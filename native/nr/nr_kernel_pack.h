#pragma once
// Content-addressed, geometry-independent CUDA code assets (not shape variants).
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
namespace dlss5::nr::detail {
using Digest=std::array<unsigned char,32>;
inline Digest digest(const void* data,size_t bytes){
    Digest result{};
    if(bytes>ULONG_MAX || BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,
        (PUCHAR)data,(ULONG)bytes,result.data(),(ULONG)result.size())<0)
        throw std::runtime_error("NR kernel SHA256 failed");
    return result;
}
inline std::string digest_hex(const Digest& d){const char* h="0123456789abcdef";std::string s;for(auto b:d){s+=h[b>>4];s+=h[b&15];}return s;}
inline std::filesystem::path environment_path(const wchar_t* name){
    DWORD n=GetEnvironmentVariableW(name,nullptr,0);if(n<=1)return {};
    std::wstring value(n,L'\0');auto used=GetEnvironmentVariableW(name,value.data(),n);
    if(!used||used>=n)return {};value.resize(used);return value;
}
struct KernelPackHeader {char magic[8];Digest source;Digest payload;uint64_t bytes;};
static_assert(sizeof(KernelPackHeader)==80);
inline std::filesystem::path kernel_pack_path(const std::filesystem::path& directory,const std::string& name,const Digest& source){
    return directory/(name+"-"+digest_hex(source)+".nrbin");
}
inline std::vector<char> read_kernel_pack(const std::filesystem::path& file,const Digest& source){
    if(!std::filesystem::exists(file))return {};
    std::ifstream f(file,std::ios::binary|std::ios::ate);auto size=f.tellg();
    KernelPackHeader h{};f.seekg(0);f.read((char*)&h,sizeof(h));
    if(!f||std::memcmp(h.magic,"D5NRGEN1",8)||h.source!=source||h.bytes==0||h.bytes>256u*1024u*1024u||size!=std::streamoff(sizeof(h)+h.bytes))
        throw std::runtime_error("Invalid generic NR kernel pack: "+file.string());
    std::vector<char> code(size_t(h.bytes));f.read(code.data(),code.size());
    if(!f||digest(code.data(),code.size())!=h.payload)throw std::runtime_error("Corrupt generic NR kernel pack: "+file.string());
    return code;
}
inline void write_kernel_pack(const std::filesystem::path& file,const Digest& source,const std::vector<char>& code){
    std::filesystem::create_directories(file.parent_path());
    KernelPackHeader h{};std::memcpy(h.magic,"D5NRGEN1",8);h.source=source;h.payload=digest(code.data(),code.size());h.bytes=code.size();
    auto temp=file;temp+=L".tmp."+std::to_wstring(GetCurrentProcessId());
    {std::ofstream f(temp,std::ios::binary|std::ios::trunc);f.write((char*)&h,sizeof(h));f.write(code.data(),code.size());if(!f)throw std::runtime_error("Cannot write generic NR kernel pack");}
    if(!MoveFileExW(temp.c_str(),file.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot publish generic NR kernel pack");
}
}
