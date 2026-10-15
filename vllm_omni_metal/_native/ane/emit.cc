// SPDX-License-Identifier: Apache-2.0
#include "ane-emitter.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc,char** argv) {
  if (argc!=6 && argc!=7) return 2;
  try {
  vpipe::AneGraphSpec s;
  s.kind=vpipe::AneGraphSpec::Kind::SwiGluFfn;
  s.M=std::stoi(argv[1]);s.K=std::stoi(argv[2]);s.N=std::stoi(argv[3]);
  s.runtime_weights=argc==7;s.weights_out_in=s.runtime_weights;s.bk=s.bn=1024;
  if (!s.valid() || s.M<1 || s.M>16384) return 2;
  if (s.K>(256LL*1024*1024)/(6LL*s.N)) return 2;
  if (s.runtime_weights) {
    std::string error;
    return vpipe::AneEmitter::emit(s,{},argv[5],&error) ? 0 : 1;
  }
  const auto expected=6ULL*s.K*s.N;
  std::error_code ec;
  const auto size=std::filesystem::file_size(argv[4],ec);
  if(ec || size!=expected || size>256ULL*1024*1024) return 2;
  std::ifstream file(argv[4],std::ios::binary);
  std::vector<unsigned char> payload{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
  std::vector<vpipe::AneBlobSlab> slabs;
  size_t offset=0;
  for(size_t bytes:vpipe::AneEmitter::weight_slabs(s)) {
    if(offset+bytes>payload.size()) return 2;
    slabs.push_back({payload.data()+offset,bytes});offset+=bytes;
  }
  if(offset!=payload.size() || file.bad()) return 2;
  std::string error;
  if(!vpipe::AneEmitter::emit(s,slabs,argv[5],&error)) {
    std::cerr<<error<<"\n";return 1;
  }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n"; return 2;
  }
}
