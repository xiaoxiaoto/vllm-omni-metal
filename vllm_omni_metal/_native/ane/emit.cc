// SPDX-License-Identifier: Apache-2.0
#include "ane-emitter.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc,char** argv) {
  const bool projection = argc == 8 && std::string(argv[6]) == "projection";
  if (argc!=6 && argc!=7 && !projection) return 2;
  try {
  vllm_omni_metal::ane::AneGraphSpec s;
  s.kind=projection ? vllm_omni_metal::ane::AneGraphSpec::Kind::Matmul : vllm_omni_metal::ane::AneGraphSpec::Kind::SwiGluFfn;
  s.M=std::stoi(argv[1]);s.K=std::stoi(argv[2]);s.N=std::stoi(argv[3]);
  s.runtime_weights=argc==7;s.weights_out_in=s.runtime_weights;s.bk=s.bn=1024;
  if (projection) {
    s.biases=std::string(argv[7])=="bias";
    s.bk=s.K%1024==0 ? 1024 : 0;
    s.bn=s.N%1024==0 ? 1024 : 0;
  }
  if (!s.valid() || s.M<1 || s.M>16384) return 2;
  const auto expected=projection ? 2ULL*s.K*s.N+(s.biases ? 2ULL*s.N : 0) : 6ULL*s.K*s.N;
  if(expected>256ULL*1024*1024) return 2;
  if (s.runtime_weights) {
    std::string error;
    return vllm_omni_metal::ane::AneEmitter::emit(s,{},argv[5],&error) ? 0 : 1;
  }
  std::error_code ec;
  const auto size=std::filesystem::file_size(argv[4],ec);
  if(ec || size!=expected || size>256ULL*1024*1024) return 2;
  std::ifstream file(argv[4],std::ios::binary);
  std::vector<unsigned char> payload{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
  std::vector<vllm_omni_metal::ane::AneBlobSlab> slabs;
  size_t offset=0;
  for(size_t bytes:vllm_omni_metal::ane::AneEmitter::weight_slabs(s)) {
    if(offset+bytes>payload.size()) return 2;
    slabs.push_back({payload.data()+offset,bytes});offset+=bytes;
  }
  if(offset!=payload.size() || file.bad()) return 2;
  std::string error;
  if(!vllm_omni_metal::ane::AneEmitter::emit(s,slabs,argv[5],&error)) {
    std::cerr<<error<<"\n";return 1;
  }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n"; return 2;
  }
}
