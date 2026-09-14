#include "image.h"
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
using Microsoft::WRL::ComPtr;
namespace d5server {
namespace {
void ok(HRESULT h){if(FAILED(h)) throw std::runtime_error("Windows image codec failed: "+std::to_string((uint32_t)h));}
struct COM { HRESULT hr=CoInitializeEx(nullptr,COINIT_MULTITHREADED); COM(){if(FAILED(hr)&&hr!=RPC_E_CHANGED_MODE)ok(hr);}~COM(){if(SUCCEEDED(hr))CoUninitialize();}};
ComPtr<IWICImagingFactory> factory(){ComPtr<IWICImagingFactory> p;ok(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&p)));return p;}
}
Image decode_image(const std::string& bytes){
 if(bytes.empty()||bytes.size()>32u*1024u*1024u)throw std::runtime_error("Image upload must be 1..32 MiB");
 COM com;auto f=factory();ComPtr<IWICStream>s;ok(f->CreateStream(&s));ok(s->InitializeFromMemory((BYTE*)bytes.data(),(DWORD)bytes.size()));
 ComPtr<IWICBitmapDecoder>d;ok(f->CreateDecoderFromStream(s.Get(),nullptr,WICDecodeMetadataCacheOnLoad,&d));ComPtr<IWICBitmapFrameDecode> frame;ok(d->GetFrame(0,&frame));
 Image image;ok(frame->GetSize(&image.width,&image.height));if(image.width<32||image.height<32||uint64_t(image.width)*image.height>16777216)throw std::runtime_error("Image axes must be >=32, at most 16 megapixels");
 ComPtr<IWICFormatConverter> cv;ok(f->CreateFormatConverter(&cv));ok(cv->Initialize(frame.Get(),GUID_WICPixelFormat32bppRGBA,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom));
 std::vector<uint8_t> raw(size_t(image.width)*image.height*4);ok(cv->CopyPixels(nullptr,image.width*4,(UINT)raw.size(),raw.data()));image.rgb.resize(raw.size());
 for(size_t i=0;i<raw.size();++i)image.rgb[i]=raw[i]*(1.f/255.f);return image;
}
std::string encode_png(const Image& image){
 COM com;auto f=factory();ComPtr<IStream> stream;ok(CreateStreamOnHGlobal(nullptr,TRUE,&stream));ComPtr<IWICBitmapEncoder> encoder;ok(f->CreateEncoder(GUID_ContainerFormatPng,nullptr,&encoder));ok(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache));
 ComPtr<IWICBitmapFrameEncode> frame;ComPtr<IPropertyBag2> props;ok(encoder->CreateNewFrame(&frame,&props));ok(frame->Initialize(props.Get()));ok(frame->SetSize(image.width,image.height));WICPixelFormatGUID format=GUID_WICPixelFormat32bppBGRA;ok(frame->SetPixelFormat(&format));
 if(!IsEqualGUID(format,GUID_WICPixelFormat32bppBGRA))throw std::runtime_error("PNG encoder does not support straight BGRA");
 if(image.channels!=4||image.rgb.size()!=size_t(image.width)*image.height*4)throw std::runtime_error("PNG requires an RGBA image");
 std::vector<uint8_t> raw(image.rgb.size());for(size_t i=0;i<raw.size();++i){float v=image.rgb[i];if(!std::isfinite(v))throw std::runtime_error("Nonfinite output");raw[i]=(uint8_t)std::nearbyint(std::clamp(v,0.f,1.f)*255.f);}
 // SetPixelFormat negotiates the accepted layout. WIC PNG uses BGRA,
 // whereas the SDK returns RGBA. Preserve straight alpha and swap only R/B.
 for(size_t i=0;i<raw.size();i+=4)std::swap(raw[i],raw[i+2]);
 ok(frame->WritePixels(image.height,image.width*4,(UINT)raw.size(),raw.data()));ok(frame->Commit());ok(encoder->Commit());STATSTG stat{};ok(stream->Stat(&stat,STATFLAG_NONAME));if(stat.cbSize.HighPart)throw std::runtime_error("PNG too large");
 HGLOBAL handle{};ok(GetHGlobalFromStream(stream.Get(),&handle));void* data=GlobalLock(handle);if(!data)throw std::runtime_error("Cannot read PNG buffer");std::string result((char*)data,stat.cbSize.LowPart);GlobalUnlock(handle);return result;
}
}
