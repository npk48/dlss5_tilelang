#include <windows.h>
#include <compressapi.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <stdexcept>
namespace fs=std::filesystem;
std::vector<unsigned char> read(const fs::path&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Cannot open "+p.u8string());auto n=f.tellg();std::vector<unsigned char>b((size_t)n);f.seekg(0);f.read((char*)b.data(),b.size());if(!f)throw std::runtime_error("Cannot read "+p.u8string());return b;}
std::array<unsigned char,32> hash(const std::vector<unsigned char>&b){std::array<unsigned char,32>h{};if(BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,(PUCHAR)b.data(),(ULONG)b.size(),h.data(),32)<0)throw std::runtime_error("SHA256 failed");return h;}
int main(int argc,char**argv){try{if(argc!=4)throw std::runtime_error("pack_dependencies OUTPUT ORT_RUNTIME TOOLCHAIN");std::vector<std::pair<fs::path,std::string>> files;
 for(auto&e:fs::directory_iterator(argv[2]))if(e.is_regular_file()&&e.path().extension()==".dll")files.emplace_back(e.path(),"bin/"+e.path().filename().u8string());
 for(auto&e:fs::recursive_directory_iterator(argv[3]))if(e.is_regular_file()){auto rel=fs::relative(e.path(),argv[3]).generic_u8string();if(e.file_size()>0&&((rel.rfind("nvrtc/bin/",0)==0&&e.path().extension()==".dll")||((rel.rfind("runtime/include/",0)==0||rel.rfind("cccl/include/",0)==0)&&e.path().extension()!=".py"&&e.path().extension()!=".pyc")))files.emplace_back(e.path(),"toolchain/"+rel);}
 std::sort(files.begin(),files.end(),[](auto&a,auto&b){return a.second<b.second;});std::ofstream out(argv[1],std::ios::binary);if(!out)throw std::runtime_error("Cannot create output");out.write("D5DEPS01",8);uint32_t count=(uint32_t)files.size();out.write((char*)&count,4);COMPRESSOR_HANDLE compressor;if(!CreateCompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&compressor))throw std::runtime_error("CreateCompressor failed");uint64_t total=0,packed=0;
 for(auto& [p,name]:files){auto b=read(p);auto h=hash(b);SIZE_T need=0;Compress(compressor,b.data(),b.size(),nullptr,0,&need);if(!need&&b.size())throw std::runtime_error("Cannot estimate compressed size");std::vector<unsigned char> compressed(need);if(!Compress(compressor,b.data(),b.size(),compressed.data(),compressed.size(),&need))throw std::runtime_error("Compression failed for "+name);uint32_t len=(uint32_t)name.size();uint64_t size=b.size(),csize=need;out.write((char*)&len,4);out.write(name.data(),len);out.write((char*)&size,8);out.write((char*)&csize,8);out.write((char*)h.data(),32);out.write((char*)compressed.data(),need);total+=size;packed+=csize;if(name.rfind("bin/",0)==0)std::cout<<name<<" "<<size<<" -> "<<csize<<"\n";}
 CloseCompressor(compressor);if(!out)throw std::runtime_error("Output write failed");std::cout<<count<<" files, "<<total<<" bytes -> "<<packed<<" bytes\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
