#include <windows.h>
#include <compressapi.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
namespace fs=std::filesystem;
#ifdef D5_EMBED_DEPENDENCIES
namespace {
std::array<unsigned char,32> digest(const unsigned char*p,size_t n){std::array<unsigned char,32>h{};if(n>0xffffffff||BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,(PUCHAR)p,(ULONG)n,h.data(),32)<0)throw std::runtime_error("Dependency checksum failed");return h;}
struct Reader{const unsigned char*p;size_t left;void read(void*out,size_t n){if(n>left)throw std::runtime_error("Truncated embedded dependencies");memcpy(out,p,n);p+=n;left-=n;}template<class T>T get(){T v;read(&v,sizeof(v));return v;}};
struct Mutex{HANDLE h=nullptr;explicit Mutex(const std::wstring&name){h=CreateMutexW(nullptr,FALSE,name.c_str());if(!h)throw std::runtime_error("Cannot lock dependency cache");auto r=WaitForSingleObject(h,600000);if(r!=WAIT_OBJECT_0&&r!=WAIT_ABANDONED){CloseHandle(h);h=nullptr;throw std::runtime_error("Timed out waiting for dependency cache");}}~Mutex(){if(h){ReleaseMutex(h);CloseHandle(h);}}};
}
#endif
void install_embedded_dependencies(){
#ifdef D5_EMBED_DEPENDENCIES
 HMODULE exe=GetModuleHandleW(nullptr);HRSRC resource=FindResourceW(exe,MAKEINTRESOURCEW(102),MAKEINTRESOURCEW(10));if(!resource)throw std::runtime_error("Embedded dependency archive missing");HGLOBAL handle=LoadResource(exe,resource);auto data=(const unsigned char*)LockResource(handle);size_t size=SizeofResource(exe,resource);if(!data||!size)throw std::runtime_error("Cannot access dependencies");auto sum=digest(data,size);std::ostringstream hex;for(auto b:sum)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)b;
 wchar_t local[32768];DWORD n=GetEnvironmentVariableW(L"LOCALAPPDATA",local,32768);if(!n||n>=32768)throw std::runtime_error("LOCALAPPDATA is unavailable");fs::path root=fs::path(local)/"DLSS5Native"/hex.str();fs::create_directories(root);Mutex lock(L"Local\\DLSS5Deps-"+fs::path(hex.str()).wstring());auto marker=root/"complete";
 if(!fs::is_regular_file(marker)){Reader reader{data,size};char magic[8];reader.read(magic,8);if(memcmp(magic,"D5DEPS01",8))throw std::runtime_error("Unknown dependency archive");uint32_t count=reader.get<uint32_t>();if(count>100000)throw std::runtime_error("Invalid dependency count");DECOMPRESSOR_HANDLE decomp;if(!CreateDecompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&decomp))throw std::runtime_error("Cannot create dependency decompressor");try{for(uint32_t i=0;i<count;++i){uint32_t len=reader.get<uint32_t>();if(!len||len>2048)throw std::runtime_error("Invalid dependency path");std::string name(len,'\0');reader.read(name.data(),len);auto relative=fs::u8path(name);if(relative.is_absolute())throw std::runtime_error("Absolute dependency path");for(auto& part:relative)if(part=="..")throw std::runtime_error("Parent dependency path");uint64_t original=reader.get<uint64_t>(),compressed=reader.get<uint64_t>();std::array<unsigned char,32>expected;reader.read(expected.data(),32);if(original>uint64_t(2)*1024*1024*1024||compressed>reader.left)throw std::runtime_error("Invalid dependency size");std::vector<unsigned char> bytes((size_t)original);SIZE_T actual=0;if(!Decompress(decomp,(PVOID)reader.p,(SIZE_T)compressed,bytes.data(),bytes.size(),&actual)||actual!=original)throw std::runtime_error("Dependency decompression failed");reader.p+=compressed;reader.left-=compressed;if(digest(bytes.data(),bytes.size())!=expected)throw std::runtime_error("Dependency checksum mismatch");auto dest=root/relative;fs::create_directories(dest.parent_path());auto tmp=dest;tmp+=L".tmp";{std::ofstream out(tmp,std::ios::binary);out.write((char*)bytes.data(),bytes.size());if(!out)throw std::runtime_error("Cannot save native dependency");}if(fs::exists(dest))fs::remove(dest);fs::rename(tmp,dest);}CloseDecompressor(decomp);}catch(...){CloseDecompressor(decomp);throw;}std::ofstream(marker)<<hex.str();}
 SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_USER_DIRS);if(!AddDllDirectory((root/"bin").c_str()))throw std::runtime_error("Cannot register native library directory");_wputenv_s(L"NATIVE_NR_TOOLCHAIN",(root/L"toolchain").c_str());
 // Establish the exact native ORT module before its first delayed C API call.
 if(!LoadLibraryExW((root/"bin"/"onnxruntime.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS))throw std::runtime_error("Cannot load packaged ONNX Runtime");
#endif
}
