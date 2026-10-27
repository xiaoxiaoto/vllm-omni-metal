// SPDX-License-Identifier: Apache-2.0
// Host-only launcher for the SAME shaders used by metal/paged_ops.cpp.
// Encode on PyTorch's stream so projections, KV writes and attention are ordered
// without a framework bridge, a CPU round trip, or a per-layer GPU wait.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <mach/mach.h>
#include <mutex>
#include <unordered_map>

#include "../metal/metal_device.h"
#include "../metal/paged_attention_kernels.h"

using at::Tensor;
namespace kernels = vllm_metal::kernels;
namespace hardware = vllm_metal::hardware;

constexpr int kPartitionSize = VLLM_METAL_PARTITION_SIZE;

static bool nax_supported() {
  auto device = at::mps::getCurrentMPSStream()->device();
  return hardware::nax_supported(
      [device supportsFamily:static_cast<MTLGPUFamily>(hardware::kApple10)]);
}

class PagedAttention {
  id<MTLLibrary> library_;
  id<MTLLibrary> nax_library_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  int gpu_cores_;
  // One bounded workspace belongs to this launcher and its first split stream.
  // Other streams use private temporaries: no resource crosses stream ownership.
  std::mutex mutex_;
  at::mps::MPSStream* scratch_stream_ = nullptr;
  Tensor scratch_tmp_, scratch_sums_, scratch_maxes_;
  static constexpr int64_t kScratchBudget = 256LL * 1024 * 1024;

  static void check_headroom(int64_t bytes) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_size;
    auto host = mach_host_self();
    auto stats_result = host_statistics64(
        host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count);
    auto page_result = host_page_size(host, &page_size);
    mach_port_deallocate(mach_task_self(), host);
    TORCH_CHECK(stats_result == KERN_SUCCESS && page_result == KERN_SUCCESS,
                "Cannot determine MPS attention system memory headroom");
    const uint64_t available =
        (static_cast<uint64_t>(vm.free_count) + vm.inactive_count) * page_size;
    TORCH_CHECK(available >= 3ULL * 1024 * 1024 * 1024 + bytes,
                "MPS attention allocation requires 3 GiB system memory headroom");
  }

  void scratch(at::mps::MPSStream* stream, const Tensor& q,
               int64_t elements, int64_t statistics, Tensor& tmp, Tensor& sums, Tensor& maxes) {
    const auto bytes = elements * q.element_size() + statistics * 8;
    TORCH_CHECK(bytes <= kScratchBudget, "MPS attention scratch exceeds 256 MiB per-call budget");
    if (scratch_stream_ == nullptr) scratch_stream_ = stream;
    const bool owner = stream == scratch_stream_;
    const bool reusable = owner && scratch_tmp_.defined() &&
        scratch_tmp_.scalar_type() == q.scalar_type() &&
        scratch_tmp_.numel() >= elements && scratch_sums_.numel() >= statistics;
    if (reusable) {
      tmp = scratch_tmp_; sums = scratch_sums_; maxes = scratch_maxes_;
      return;
    }
    check_headroom(bytes);
    tmp = at::empty({elements}, q.options());
    sums = at::empty({statistics}, q.options().dtype(at::kFloat));
    maxes = at::empty_like(sums);
    if (owner) {
      // Keep only the latest capacity, never a shape-indexed unbounded cache.
      // Previously encoded Metal commands retain old buffers through completion.
      scratch_tmp_ = tmp; scratch_sums_ = sums; scratch_maxes_ = maxes;
    }
  }

  id<MTLComputePipelineState> pipeline(const std::string& name, bool split = false,
                                      int window_q = 0) {
    const auto key = name + "_wq" + std::to_string(window_q);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    auto constants = [[MTLFunctionConstantValues alloc] init];
    bool no = false;
    for (NSUInteger i : {20, 30, 40, 50, 100, 120, 130})
      [constants setConstantValue:&no type:MTLDataTypeBool atIndex:i];
    [constants setConstantValue:&split type:MTLDataTypeBool atIndex:10];
    int kbits = 8, vbits = 3;
    [constants setConstantValue:&kbits type:MTLDataTypeInt atIndex:60];
    [constants setConstantValue:&vbits type:MTLDataTypeInt atIndex:70];
    [constants setConstantValue:&window_q type:MTLDataTypeInt atIndex:110];
    auto lib = name.find("paged_attention_nax_") == 0 ? nax_library_ : library_;
    NSError* error = nil;
    auto fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                       constantValues:constants error:&error];
    [constants release];
    TORCH_CHECK(fn, "MPS attention function ", name, ": ",
                error ? error.localizedDescription.UTF8String : "missing function");
    auto pso = [at::mps::getCurrentMPSStream()->device()
        newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    TORCH_CHECK(pso, "MPS attention pipeline: ", error.localizedDescription.UTF8String);
    pipelines_[key] = pso;
    return pso;
  }

  static void buffer(id<MTLComputeCommandEncoder> enc, int i, const Tensor& t) {
    [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
            offset:t.storage_offset() * t.element_size() atIndex:i];
  }
  template<typename T>
  static void scalar(id<MTLComputeCommandEncoder> enc, int i, T value) {
    [enc setBytes:&value length:sizeof(T) atIndex:i];
  }

 public:
  PagedAttention(const std::string& path, const std::string& nax_path, int gpu_cores)
      : gpu_cores_(gpu_cores) {
    @autoreleasepool {
      auto device = at::mps::getCurrentMPSStream()->device();
      NSError* error = nil;
      library_ = [device newLibraryWithURL:[NSURL fileURLWithPath:
          [NSString stringWithUTF8String:path.c_str()]] error:&error];
      TORCH_CHECK(library_, "Cannot load paged attention metallib: ",
                  error.localizedDescription.UTF8String);
      if (!nax_path.empty()) {
        nax_library_ = [device newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:nax_path.c_str()]] error:&error];
        if (!nax_library_)
          TORCH_WARN("Cannot load NAX metallib; using the non-NAX fallback: ",
                     error.localizedDescription.UTF8String);
      }
    }
  }
  ~PagedAttention() {
    for (auto& p : pipelines_) [p.second release];
    [library_ release];
    [nax_library_ release];
  }

  void forward(const Tensor& q, const Tensor& k, const Tensor& v,
               const Tensor& kc, const Tensor& vc, const Tensor& slots,
               const Tensor& blocks, const Tensor& lens, const Tensor& cu,
               const Tensor& out, int max_seq_len, double scale,
               int sliding_window, double softcap, int window_seqlen_q) {
    // Serialize the entire encode transaction (including pipeline/cache mutation).
    // This is a CPU lock, not a GPU wait; command order protects same-stream reuse.
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* t : {&q, &k, &v, &kc, &vc, &slots, &blocks, &lens, &cu, &out})
      TORCH_CHECK(t->is_mps(), "All paged attention buffers must be on MPS");
    for (auto* t : {&q, &k, &v, &kc, &vc, &slots, &blocks, &lens, &cu})
      TORCH_CHECK(!out.is_alias_of(*t),
                  "Attention output must not share storage with inputs or KV cache");
    TORCH_CHECK(!out.requires_grad() && !out.is_conj() && !out.is_neg(),
                "Attention output must be a plain tensor without autograd");
    TORCH_CHECK(q.dim() == 3 && q.is_contiguous() &&
                (q.size(2) == 64 || q.size(2) == 96 || q.size(2) == 128 ||
                 q.size(2) == 256 || q.size(2) == 512),
                "MPS attention requires contiguous Q with head_dim=64/96/128/256/512");
    const int head_size = q.size(2);
    TORCH_CHECK(kc.dim() == 4 &&
                (kc.size(1) == 8 || kc.size(1) == 16 || kc.size(1) == 32) &&
                kc.size(3) == head_size &&
                kc.stride(3) == 1 && vc.strides() == kc.strides() && kc.sizes() == vc.sizes(),
                "MPS attention requires [blocks,8/16/32,kv_heads,head_dim] KV");
    const int block_size = kc.size(1);
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16 ||
                q.scalar_type() == at::kFloat,
                "MPS attention requires fp32, fp16 or bf16");
    for (auto* t : {&k, &v, &kc, &vc, &out})
      TORCH_CHECK(t->scalar_type() == q.scalar_type(), "Q/K/V/cache/output dtype mismatch");
    TORCH_CHECK(slots.scalar_type() == at::kLong && blocks.scalar_type() == at::kInt &&
                lens.scalar_type() == at::kInt && cu.scalar_type() == at::kInt,
                "Invalid attention metadata dtype");
    TORCH_CHECK(slots.is_contiguous() && blocks.is_contiguous() &&
                lens.is_contiguous() && cu.is_contiguous(),
                "Attention metadata must be contiguous");
    const int tokens = q.size(0), heads = q.size(1), kv_heads = kc.size(2);
    const int seqs = lens.numel();
    TORCH_CHECK(max_seq_len > 0 && tokens > 0 && heads > 0 && kv_heads > 0 && seqs > 0,
                "Attention dimensions and maximum sequence length must be positive");
    const int parts = (static_cast<int64_t>(max_seq_len) + kPartitionSize - 1) / kPartitionSize;
    const int64_t statistics = static_cast<int64_t>(tokens) * heads * parts;
    TORCH_CHECK(statistics <= kScratchBudget / (head_size * q.element_size() + 8),
                "MPS attention scratch exceeds 256 MiB per-call budget");
    TORCH_CHECK(k.sizes() == v.sizes() && k.dim() == 3 && k.size(0) == tokens &&
                k.size(1) == kv_heads && k.size(2) == head_size &&
                k.stride(2) == 1 && k.stride(1) == head_size &&
                v.stride(2) == 1 && v.stride(1) == head_size &&
                out.sizes() == q.sizes() && out.is_contiguous() &&
                slots.numel() == tokens && blocks.dim() == 2 && blocks.size(0) == seqs &&
                cu.numel() == seqs + 1 && heads % kv_heads == 0,
                "Inconsistent attention shapes/strides");
    // Reuse the same short verification-window shader mode as the MLX launcher.
    const bool window = tokens > seqs && window_seqlen_q > 1 &&
        head_size <= VLLM_METAL_PA_WINDOW_MAX_HEAD;
    const int window_q = window ?
        (window_seqlen_q + VLLM_METAL_PA_WINDOW_ROWS - 1) / VLLM_METAL_PA_WINDOW_ROWS : 0;
    // The released tiled kernels only cover half/bfloat; float uses the same paged kernel.
    const bool prefill = tokens > seqs && !window && q.scalar_type() != at::kFloat;
    const bool nax = prefill && nax_library_ != nil;
    const bool split = !prefill
        && kernels::should_split_decode(heads, tokens, parts, gpu_cores_);
    const std::string dt = q.scalar_type() == at::kHalf ? "half" :
        q.scalar_type() == at::kBFloat16 ? "bfloat16_t" : "float";
    const auto cfg = kernels::select_tile_config(head_size).value();
    auto scatter = pipeline(kernels::reshape_and_cache_name(dt, dt));
    auto name = nax ? kernels::nax_name(dt, head_size, block_size) :
        prefill ? kernels::tiled_name(dt, head_size, block_size, cfg) :
        kernels::paged_name(
            dt, dt, dt, head_size, block_size, split ? kPartitionSize : 0);
    auto attn = pipeline(name, split, window_q);
    id<MTLComputePipelineState> reduce = nil;
    auto stream = at::mps::getCurrentMPSStream();
    Tensor tmp, sums, maxes;
    if (split) {
      scratch(stream, q, statistics * head_size, statistics, tmp, sums, maxes);
      reduce = pipeline(
          kernels::paged_reduce_kernel_name(dt, head_size, kPartitionSize));
    }
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
      @autoreleasepool {
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:scatter];
        buffer(enc, 0, k); buffer(enc, 1, v);
        buffer(enc, 2, kc); buffer(enc, 3, vc); buffer(enc, 4, slots);
        scalar<int>(enc, 7, k.stride(0)); scalar<int>(enc, 8, v.stride(0));
        scalar<int>(enc, 9, kv_heads); scalar<int>(enc, 10, head_size);
        scalar<int>(enc, 11, block_size);
        scalar<int64_t>(enc, 12, kc.stride(0));
        scalar<int64_t>(enc, 13, kc.stride(1));
        scalar<int64_t>(enc, 14, kc.stride(2));
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:attn];
        buffer(enc, 2, split ? tmp : out); buffer(enc, 3, q);
        buffer(enc, 4, kc); buffer(enc, 5, vc);
        scalar<int>(enc, 8, kv_heads); scalar<float>(enc, 9, scale);
        scalar<float>(enc, 10, softcap);
        buffer(enc, 11, blocks); buffer(enc, 12, lens);
        scalar<int>(enc, 13, blocks.size(1));
        scalar<int>(enc, 15, heads * head_size);
        scalar<int>(enc, 16, kc.stride(0)); scalar<int>(enc, 17, kc.stride(2));
        buffer(enc, 19, cu); scalar<int>(enc, 20, seqs);
        scalar<int>(enc, 21, sliding_window);
        int grid_y = window ? seqs * window_q : tokens, threads = 256;
        if (prefill) {
          grid_y = tokens / (nax ? kernels::kNaxBQ : cfg.BQ) + seqs;
          threads = nax ? kernels::kNaxThreads : cfg.NUM_THREADS;
          scalar<int>(enc, 30, 0);  // This launch starts at the first query block.
          if (!nax) [enc setThreadgroupMemoryLength:
              kernels::tiled_shared_bytes(cfg, head_size, q.element_size()) atIndex:0];
        } else {
          [enc setThreadgroupMemoryLength:kernels::paged_threadgroup_bytes(
              head_size, block_size, static_cast<int>(q.element_size()),
              window, VLLM_METAL_PA_WINDOW_ROWS) atIndex:0];
        }
        if (split) { buffer(enc, 0, sums); buffer(enc, 1, maxes); }
        [enc dispatchThreadgroups:MTLSizeMake(heads, grid_y, split ? parts : 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        if (split) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [enc setComputePipelineState:reduce];
          [enc setThreadgroupMemoryLength:
              kernels::paged_reduce_threadgroup_bytes(parts) atIndex:0];
          buffer(enc, 0, out); buffer(enc, 1, sums); buffer(enc, 2, maxes);
          buffer(enc, 3, tmp); buffer(enc, 4, lens); scalar<int>(enc, 5, parts);
          buffer(enc, 7, cu); scalar<int>(enc, 8, seqs);
          [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          // The next call may overwrite the workspace on this encoder. Order its
          // writes after this reducer's reads without waiting for GPU completion.
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
      }
    });
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("detected_gpu_core_count", &hardware::detected_gpu_core_count);
  m.def("gpu_core_count", &hardware::gpu_core_count);
  m.def("_override_detected_gpu_core_count_for_test",
        &hardware::override_detected_gpu_core_count_for_test);
  m.def("nax_supported", &nax_supported);
  pybind11::class_<PagedAttention>(m, "PagedAttention")
      .def(pybind11::init<const std::string&, const std::string&, int>())
      .def("forward", &PagedAttention::forward);
}
