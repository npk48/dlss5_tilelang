#include "pipeline.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace dlss5::pipeline {
namespace {
void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(std::string("pipeline CUDA: ")+cudaGetErrorString(e)); }
void require(bool ok, const char* message) { if(!ok) throw std::invalid_argument(message); }
void validate(const D5Tensor& t, unsigned channels=0, unsigned layout=D5_HWC) {
    require(t.struct_size>=sizeof(D5Tensor) && t.data && t.dtype==D5_F32 && t.layout==layout && t.width && t.height && t.channels,
            "pipeline requires nonempty device F32 tensor with declared layout");
    require(!channels || t.channels==channels,"pipeline tensor channel mismatch");
    uint64_t row=uint64_t(t.width)*(layout==D5_HWC?t.channels:1)*4;
    require(!t.row_stride_bytes || (t.row_stride_bytes>=row && t.row_stride_bytes%4==0),"invalid row stride");
    require(!t.plane_stride_bytes || (t.plane_stride_bytes>=(t.row_stride_bytes?t.row_stride_bytes:row)*t.height && t.plane_stride_bytes%4==0),"invalid plane stride");
    require(t.data%4==0 && uint64_t(t.width)*t.height<0x7fffffffULL,"unaligned address or unsupported extent");
}
void same(const D5Tensor& a,const D5Tensor& b) { require(a.width==b.width && a.height==b.height,"pipeline tensor extent mismatch"); }
struct View {
    float* p; int w,h,c; size_t row,plane; bool chw;
    __device__ float get(int y,int x,int k) const { return p[size_t(y)*row+(chw?size_t(k)*plane+x:size_t(x)*c+k)]; }
    __device__ void put(int y,int x,int k,float f) const { p[size_t(y)*row+(chw?size_t(k)*plane+x:size_t(x)*c+k)]=f; }
};
View v(const D5Tensor& t) { size_t row=t.row_stride_bytes?t.row_stride_bytes/4:size_t(t.width)*(t.layout==D5_CHW?1:t.channels); return {reinterpret_cast<float*>(t.data),int(t.width),int(t.height),int(t.channels),row,t.plane_stride_bytes?t.plane_stride_bytes/4:row*t.height,t.layout==D5_CHW}; }
View optional(const D5Tensor& t) { return t.data?v(t):View{}; }
__device__ float sat(float x) { return isnan(x)?x:fminf(fmaxf(x,0.f),1.f); }
__device__ float rtz(float x) { return __half2float(__float2half_rz(x)); }
__device__ float rn(float x) { return __half2float(__float2half_rn(x)); }
__device__ float fetch(View a,int y,int x,int c,bool border) {
    if(border) return a.get(min(max(y,0),a.h-1),min(max(x,0),a.w-1),c);
    return x<0 || y<0 || x>=a.w || y>=a.h?0.f:a.get(y,x,c);
}
__device__ float linear_xy(View a,float x,float y,int c,bool border) {
    if(border) { x=fminf(fmaxf(x,0.f),float(a.w-1)); y=fminf(fmaxf(y,0.f),float(a.h-1)); }
    int ix=int(floorf(x)),iy=int(floorf(y));
    float nw=(float(ix+1)-x)*(float(iy+1)-y),ne=(x-float(ix))*(float(iy+1)-y);
    float sw=(float(ix+1)-x)*(y-float(iy)),se=(x-float(ix))*(y-float(iy));
    // grid_sample's CUDA kernel uses four ordered FMA accumulations even
    // though helper arithmetic elsewhere is built with --fmad=false.
    float out=0.f;
    out=__fmaf_rn(fetch(a,iy,ix,c,border),nw,out);
    out=__fmaf_rn(fetch(a,iy,ix+1,c,border),ne,out);
    out=__fmaf_rn(fetch(a,iy+1,ix,c,border),sw,out);
    return __fmaf_rn(fetch(a,iy+1,ix+1,c,border),se,out);
}
__device__ float texture(View a,float u,float vv,int c,bool border) {
    float gx=u*2.f-1.f,gy=vv*2.f-1.f;
    float x=__fmaf_rn(gx+1.f,float(a.w),-1.f)*.5f,y=__fmaf_rn(gy+1.f,float(a.h),-1.f)*.5f;
    return linear_xy(a,x,y,c,border);
}
__device__ float mapped_texture(View a,float px,float py,int c) {
    float u=px*(1.f/a.w),vv=py*(1.f/a.h);
    u=(u*a.w)/float(a.w); vv=(vv*a.h)/float(a.h);
    return texture(a,u,vv,c,true);
}
__device__ void weights(float t,float* w) {
    float t2=t*t,t3=t2*t;
    w[0]=-.5f*t+t2-.5f*t3; w[1]=1.f-2.5f*t2+1.5f*t3;
    w[2]=.5f*t+2.f*t2-1.5f*t3; w[3]=-.5f*t2+.5f*t3;
}
__device__ float catmull(View a,float u,float vv,int c) {
    float ix=floorf(u*a.w-.5f),iy=floorf(vv*a.h-.5f),wx[4],wy[4];
    weights(sat(u*a.w-(ix+.5f)),wx); weights(sat(vv*a.h-(iy+.5f)),wy);
    float gx=wx[1]+wx[2],gy=wy[1]+wy[2];
    float xl=fminf(fmaxf(ix-.5f,.5f),a.w-.5f),xr=fminf(fmaxf(ix+2.5f,.5f),a.w-.5f);
    float xc=fminf(fmaxf(ix+.5f+wx[2]/fmaxf(gx,0x1p-23f),.5f),a.w-.5f);
    float yt=fminf(fmaxf(iy-.5f,.5f),a.h-.5f),yb=fminf(fmaxf(iy+2.5f,.5f),a.h-.5f);
    float yc=fminf(fmaxf(iy+.5f+wy[2]/fmaxf(gy,0x1p-23f),.5f),a.h-.5f);
    float wt=gx*wy[0],wl=wx[0]*gy,wc=gx*gy,wb=gx*wy[3],wr=wx[3]*gy;
    float out=mapped_texture(a,xc,yt,c)*wt+mapped_texture(a,xl,yc,c)*wl;
    out=out+mapped_texture(a,xc,yc,c)*wc;
    out=out+mapped_texture(a,xc,yb,c)*wb;
    out=out+mapped_texture(a,xr,yc,c)*wr;
    return out/fmaxf(wt+wl+wc+wb+wr,0x1p-23f);
}
__device__ unsigned pcg(unsigned q) { return ((q>>((q>>28)+4))^q)*0x108EF2D9u; }
__device__ float uniform24(unsigned q) { unsigned r=pcg(q); return float(((r>>30)^(r>>8))+1u)*0x1p-24f; }
__device__ void gaussian(int x,int y,unsigned frame,float* g) {
    unsigned q=(unsigned(x)*0x8DA6B343u)^(unsigned(y)*0xD8163841u)^(frame*0x9E3779B9u)^0x243F6A88u;
    unsigned r=pcg(q),h=(r>>22)^r;
    float u0=uniform24(h*0xCAA5B80Du+0x21DD796Bu),u1=uniform24(h*0x83232C31u+0x3463E0ACu);
    float u2=uniform24(h*0x2C9277B5u+0xAC564B05u),u3=uniform24(h*0xFA6DC5F9u+0x4712A88Eu);
    float r0=sqrtf(-2.f*logf(u0)),r1=sqrtf(-2.f*logf(u2));
    float a0=6.283185307179586f*u1,a1=6.283185307179586f*u3;
    g[0]=rn(r0*cosf(a0)); g[1]=rn(r1*cosf(a1)); g[2]=rn(r1*sinf(a1));
}
__global__ void prepare_k(View work,View motion,View conf,View current,View mv,View gate,bool cold,float strength,float mx,float my) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=current.w*current.h)return;
    int y=i/current.w,x=i%current.w;
    int sy=max(0,min(y<work.h?y:2*work.h-2-y,work.h-1)),sx=max(0,min(x<work.w?x:2*work.w-2-x,work.w-1));
    for(int k=0;k<3;k++)current.put(y,x,k,rtz(work.get(sy,sx,k)));
    float dx=0.f,dy=0.f,g=0.f;
    if(!cold && y<work.h && x<work.w) {
        int iy=min(int(floorf(((float(y)+.5f)*motion.h)*(1.f/work.h))),motion.h-1);
        int ix=min(int(floorf(((float(x)+.5f)*motion.w)*(1.f/work.w))),motion.w-1);
        // Preserve normalized-UV then work-pixel arithmetic of prepare_kernel.
        dx=(motion.get(iy,ix,0)*mx)*float(work.w);dy=(motion.get(iy,ix,1)*my)*float(work.h);
        float px=float(x)+.5f+dx,py=float(y)+.5f+dy;
        bool visible=px>=0.f && px<=work.w && py>=0.f && py<=work.h;
        float confidence=conf.p?linear_xy(conf,(float(x)+.5f)*(float(conf.w)/work.w)-.5f,(float(y)+.5f)*(float(conf.h)/work.h)-.5f,0,true):1.f;
        g=(float(visible)*confidence)*strength;
    }
    mv.put(y,x,0,dx);mv.put(y,x,1,dy);gate.put(y,x,0,g);
}
struct Scalars { float tone,structure,skin,automatic,style; };
__global__ void packet_k(View current,View old,View mv,View gate,View sampled,View warped,View packet,unsigned frame,Scalars s) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=current.w*current.h)return;
    int y=i/current.w,x=i%current.w;
    float u=(float(x)+.5f)*(1.f/current.w),vv=(float(y)+.5f)*(1.f/current.h);
    float cu=(u*current.w)/float(current.w),cv=(vv*current.h)/float(current.h);
    float hu=u+texture(mv,cu,cv,0,false)*(1.f/current.w),hv=vv+texture(mv,cu,cv,1,false)*(1.f/current.h);
    float g[3];gaussian(x,y,frame,g);
    packet.put(y,x,0,g[1]);packet.put(y,x,1,g[2]);packet.put(y,x,2,g[0]);packet.put(y,x,3,1.f);
    for(int k=0;k<3;k++) {
        float c=texture(current,cu,cv,k,false),p=catmull(old,hu,hv,k);
        sampled.put(y,x,k,c);warped.put(y,x,k,p);
        float cc=rn(rn(rn(c)-.5f)*.125f),pp=rn(rn(rn(p)-.5f)*.125f);
        packet.put(y,x,k+4,cc);packet.put(y,x,k+7,gate.get(y,x,0)>0.f?pp:cc);
    }
    packet.put(y,x,10,s.style);packet.put(y,x,11,s.tone);packet.put(y,x,12,s.structure);
    packet.put(y,x,13,s.skin);packet.put(y,x,14,s.automatic);packet.put(y,x,15,0.f);
}
__global__ void composite_k(View current,View sampled,View previous,View gate,View head,View result,View next,float intensity) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=current.w*current.h)return;
    int y=i/current.w,x=i%current.w;float sigmoid=1.f/(1.f+expf(-head.get(y,x,3)));
    float alpha=sat(sigmoid*sat(gate.get(y,x,0)));
    for(int k=0;k<3;k++) {
        float centered=sampled.get(y,x,k)*.125f-.0625f;
        float corrected=sat((centered+head.get(y,x,k)*.03125f)*8.f+.5f);
        float r=corrected+alpha*(previous.get(y,x,k)-corrected);
        if(intensity!=1.f)r=current.get(y,x,k)+intensity*(r-current.get(y,x,k));
        result.put(y,x,k,r);next.put(y,x,k,rtz(r));
    }
}
__device__ float to_linear(float a) { a=fmaxf(a,0.f);return a<=.04045f?a/12.92f:powf((a+.055f)/1.055f,2.4f); }
__device__ float to_srgb(float a) { a=fmaxf(a,0.f);return a<=.0031308f?12.92f*a:1.055f*powf(a,1.f/2.4f)-.055f; }
__device__ float pq_nits(float a) { float p=powf(sat(a),1.f/(2523.f/32.f));return 10000.f*powf(fmaxf(p-3424.f/4096.f,0.f)/(2413.f/128.f-(2392.f/128.f)*p),1.f/(2610.f/16384.f)); }
__device__ float nits_pq(float a) { float p=powf(fminf(fmaxf(a,0.f),10000.f)/10000.f,2610.f/16384.f);return powf((3424.f/4096.f+(2413.f/128.f)*p)/(1.f+(2392.f/128.f)*p),2523.f/32.f); }
__device__ void transform(float* a,bool to709) {
    float r=a[0],g=a[1],b=a[2];
    if(to709) { a[0]=1.660491f*r-.587641f*g-.072850f*b;a[1]=-.124550f*r+1.132900f*g-.008349f*b;a[2]=-.018151f*r-.100579f*g+1.118730f*b; }
    else { a[0]=.627404f*r+.329283f*g+.043313f*b;a[1]=.069097f*r+.919540f*g+.011362f*b;a[2]=.016391f*r+.088013f*g+.895595f*b; }
}
bool hdr(const ColorOptions& c) { return c.input_encoding>=D5_COLOR_LINEAR709_NITS; }
void color_options(const ColorOptions& c) {
    require(c.input_encoding<=D5_COLOR_PQ2020 && c.output_encoding<=D5_COLOR_PQ2020,"unknown color encoding");
    require(hdr(c)==(c.output_encoding>=D5_COLOR_LINEAR709_NITS),"HDR/SDR output mode mismatch");
    require(std::isfinite(c.reference_white_nits) && c.reference_white_nits>0 && std::isfinite(c.peak_nits) && c.peak_nits>0 && c.peak_nits<=10000,"invalid white or peak nits");
}
__global__ void decode_k(View source,View target,unsigned encoding) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=target.w*target.h)return;int y=i/target.w,x=i%target.w;
    float a[3];for(int k=0;k<3;k++){a[k]=source.get(y,x,k);if(encoding==D5_COLOR_SRGB)a[k]=to_linear(a[k]);else if(encoding==D5_COLOR_PQ2020)a[k]=pq_nits(a[k]);}
    if(encoding==D5_COLOR_PQ2020)transform(a,true);for(int k=0;k<3;k++)target.put(y,x,k,a[k]);
}
__global__ void encode_k(View source,View target,bool is_hdr,float white) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=target.w*target.h)return;int y=i/target.w,x=i%target.w;
    float a[3];float high=0.f;for(int k=0;k<3;k++){a[k]=source.get(y,x,k);if(is_hdr)a[k]=fmaxf(a[k],0.f)/white;high=fmaxf(high,a[k]);}
    for(int k=0;k<3;k++)target.put(y,x,k,is_hdr?to_srgb(a[k]/(1.f+high)):sat(to_srgb(a[k])));
}
__device__ void bridge_decode(View a,int x,int y,bool is_hdr,float white,float peak,float* rgb) {
    float high=0.f;for(int k=0;k<3;k++){rgb[k]=to_linear(sat(a.get(y,x,k)));high=fmaxf(high,rgb[k]);}
    if(is_hdr){float p=peak/white,scale=fminf(1.f/fmaxf(1.f-high,1.f/(1.f+p)),p/(high>0.f?high:1.f));for(int k=0;k<3;k++)rgb[k]=(rgb[k]*scale)*white;}
}
__global__ void finish_k(View context,View reference,View modified,View original,View protect,View output,bool is_hdr,float white,float peak,unsigned encoding,float mix) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=output.w*output.h)return;int y=i/output.w,x=i%output.w;
    float alpha=original.c==4?linear_xy(original,(float(x)+.5f)*(float(original.w)/output.w)-.5f,(float(y)+.5f)*(float(original.h)/output.h)-.5f,3,true):1.f;
    float strength=mix;if(protect.p)strength*=1.f-protect.get(y,x,0);strength*=float(alpha>0.f);
    float ref[3],mod[3],out[3];bridge_decode(reference,x,y,is_hdr,white,peak,ref);bridge_decode(modified,x,y,is_hdr,white,peak,mod);
    for(int k=0;k<3;k++){float ctx=context.get(y,x,k);float candidate=ctx+(mod[k]-ref[k]);float delta=candidate-ctx;out[k]=strength==0.f?ctx:ctx+delta*strength;}
    if(encoding==D5_COLOR_SRGB)for(int k=0;k<3;k++)out[k]=sat(to_srgb(out[k]));
    else if(encoding==D5_COLOR_PQ2020){transform(out,false);for(int k=0;k<3;k++)out[k]=nits_pq(out[k]);}
    for(int k=0;k<3;k++)output.put(y,x,k,out[k]);if(output.c==4)output.put(y,x,3,alpha);
}
__global__ void resize_k(View src,View dst,bool aa) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dst.w*dst.h)return;int y=i/dst.w,x=i%dst.w;
    float sx=float(src.w)/dst.w,sy=float(src.h)/dst.h;
    float cx=sx*(float(x)+.5f),cy=sy*(float(y)+.5f);
    if(!aa){for(int k=0;k<dst.c;k++)dst.put(y,x,k,linear_xy(src,cx-.5f,cy-.5f,k,true));return;}
    float ax=fmaxf(sx,1.f),ay=fmaxf(sy,1.f);
    int x0=max(int(cx-ax+.5f),0),x1=min(int(cx+ax+.5f),src.w);
    int y0=max(int(cy-ay+.5f),0),y1=min(int(cy+ay+.5f),src.h);
    float sumx=0.f,sumy=0.f;
    for(int xx=x0;xx<x1;xx++)sumx+=fmaxf(0.f,1.f-fabsf((float(xx)-cx+.5f)/ax));
    for(int yy=y0;yy<y1;yy++)sumy+=fmaxf(0.f,1.f-fabsf((float(yy)-cy+.5f)/ay));
    for(int k=0;k<dst.c;k++) { float result=0.f;
        for(int yy=y0;yy<y1;yy++){float row=0.f;for(int xx=x0;xx<x1;xx++){float w=fmaxf(0.f,1.f-fabsf((float(xx)-cx+.5f)/ax))/sumx;row=__fmaf_rn(src.get(yy,xx,k),w,row);}float w=fmaxf(0.f,1.f-fabsf((float(yy)-cy+.5f)/ay))/sumy;result=__fmaf_rn(row,w,result);}
        dst.put(y,x,k,result);
    }
}
__global__ void half_k(View src,View dst) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dst.w*dst.h)return;int y=i/dst.w,x=i%dst.w;for(int k=0;k<dst.c;k++)dst.put(y,x,k,rtz(src.get(y,x,k)));}
__global__ void delta_k(View result,View base,View delta) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=delta.w*delta.h)return;int y=i/delta.w,x=i%delta.w;for(int k=0;k<3;k++)delta.put(y,x,k,result.get(y,x,k)-base.get(y,x,k));}
__global__ void transport_k(View reference,View delta,View dst) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dst.w*dst.h)return;int y=i/dst.w,x=i%dst.w;for(int k=0;k<3;k++)dst.put(y,x,k,sat(reference.get(y,x,k)+linear_xy(delta,(float(x)+.5f)*(float(delta.w)/dst.w)-.5f,(float(y)+.5f)*(float(delta.h)/dst.h)-.5f,k,true)));}
unsigned blocks(const D5Tensor& t) {return unsigned((uint64_t(t.width)*t.height+127)/128);}
D5Tensor tensor(float* p,unsigned w,unsigned h,unsigned c,unsigned layout=D5_HWC) {return {sizeof(D5Tensor),D5_F32,layout,c,w,h,reinterpret_cast<uint64_t>(p),0,0};}
}

bool Ops::supports(Reconstruction type) noexcept {return type==Reconstruction::Bilinear;}
void Ops::preprocess(const D5Tensor& source,const D5Tensor& context,const ColorOptions& c,CUstream stream) {validate(source);require(source.channels==3 || source.channels==4,"source must be RGB(A)");validate(context,3);same(source,context);color_options(c);decode_k<<<blocks(context),128,0,stream>>>(v(source),v(context),c.input_encoding);check(cudaGetLastError());}
void Ops::encode(const D5Tensor& context,const D5Tensor& encoded,const ColorOptions& c,CUstream stream) {validate(context,3);validate(encoded,3);same(context,encoded);color_options(c);encode_k<<<blocks(encoded),128,0,stream>>>(v(context),v(encoded),hdr(c),c.reference_white_nits);check(cudaGetLastError());}
void Ops::resize(const D5Tensor& source,const D5Tensor& target,CUstream stream,bool antialias) {validate(source);validate(target,source.channels);if(source.data==target.data){same(source,target);return;}resize_k<<<blocks(target),128,0,stream>>>(v(source),v(target),antialias);check(cudaGetLastError());}
void Ops::reconstruct(const D5Tensor& source,const D5Tensor& target,Reconstruction type,CUstream stream) {if(!supports(type))throw std::runtime_error("requested FSR2/EASU reconstruction is unsupported by native pipeline; no replacement is applied");resize(source,target,stream,true);}
void Ops::half_rtz(const D5Tensor& source,const D5Tensor& target,CUstream stream) {validate(source);validate(target,source.channels);same(source,target);half_k<<<blocks(target),128,0,stream>>>(v(source),v(target));check(cudaGetLastError());}
void Ops::transport(const D5Tensor& reference,const D5Tensor& result,const D5Tensor& base,const D5Tensor& target,CUstream stream) {
    validate(reference,3);validate(result,3);validate(base,3);validate(target,3);same(reference,target);same(result,base);
    float* data=nullptr;check(cudaMallocAsync(&data,size_t(result.width)*result.height*3*4,stream));D5Tensor delta=tensor(data,result.width,result.height,3);
    delta_k<<<blocks(delta),128,0,stream>>>(v(result),v(base),v(delta));transport_k<<<blocks(target),128,0,stream>>>(v(reference),v(delta),v(target));cudaError_t e=cudaGetLastError();cudaError_t f=cudaFreeAsync(data,stream);check(e);check(f);
}
void Ops::finish(const D5Tensor& context,const D5Tensor& reference,const D5Tensor& modified,const D5Tensor& original,const D5Tensor& protect,const D5Tensor& output,const ColorOptions& c,float mix,CUstream stream) {
    validate(context,3);validate(reference,3);validate(modified,3);validate(original);validate(output);require((original.channels==3 || original.channels==4)&&(output.channels==3 || output.channels==4),"finish requires RGB(A)");same(context,reference);same(context,modified);same(context,output);if(protect.data){validate(protect,1);same(protect,output);}color_options(c);require(std::isfinite(mix)&&mix>=0.f&&mix<=1.f,"mix must be [0,1]");
    finish_k<<<blocks(output),128,0,stream>>>(v(context),v(reference),v(modified),v(original),optional(protect),v(output),hdr(c),c.reference_white_nits,c.peak_nits,c.output_encoding,mix);check(cudaGetLastError());
}

struct PipelineState::Impl {
    unsigned w=0,h=0,nw=0,nh=0,frame=0,pending_frame=0;bool valid=false,prepared=false,composited=false;CUstream stream=nullptr;D5NRSettings settings{};
    float* data=nullptr;D5Tensor current{},mv{},gate{},sampled{},warped{},packet{},result{},old{},next{};
    ~Impl(){if(data)cudaFree(data);}
};
PipelineState::PipelineState():p_(new Impl){}
PipelineState::~PipelineState()=default;
void PipelineState::allocate(uint32_t w,uint32_t h) {
    require(w>=32 && h>=32 && w<=32768 && h<=32768,"NR work extent must be 32..32768");
    if(p_->w==w && p_->h==h)return;
    auto q=std::make_unique<Impl>();q->w=w;q->h=h;q->nw=std::max(320u,((w+127)/64)*64);q->nh=std::max(320u,((h+63)/64)*64);
    size_t n=size_t(q->nw)*q->nh;check(cudaMalloc(&q->data,n*37*4));float* d=q->data;
    auto take=[&](unsigned c,unsigned layout=D5_HWC){auto t=tensor(d,q->nw,q->nh,c,layout);d+=n*c;return t;};
    q->current=take(3);q->mv=take(2);q->gate=take(1);q->sampled=take(3);q->warped=take(3);q->packet=take(16,D5_CHW);q->result=take(3);q->old=take(3);q->next=take(3);p_=std::move(q);
}
void PipelineState::reset() noexcept {p_->valid=false;p_->prepared=false;p_->composited=false;p_->frame=0;}
void PipelineState::prepare(const D5Tensor& work,const D5Tensor& motion,const D5Tensor& conf,const D5NRSettings& s,bool reset_flag,CUstream stream,uint32_t mw,uint32_t mh) {
    auto& p=*p_;validate(work,3);require(p.data && work.width==p.w && work.height==p.h,"allocate state for work extents before prepare");
    require(s.struct_size>=sizeof(D5NRSettings)&&s.style<=255&&s.automatic_mask<=1,"invalid NR settings");
    require(std::isfinite(s.structure)&&std::isfinite(s.skin)&&std::isfinite(s.tone)&&std::isfinite(s.temporal_strength)&&std::isfinite(s.intensity)&&s.temporal_strength>=0&&s.temporal_strength<=1&&s.intensity>=0&&s.intensity<=1,"invalid NR scalar");
    if(motion.data)validate(motion,2);if(conf.data)validate(conf,1);
    p.prepared=false;p.composited=false;p.stream=stream;p.settings=s;p.pending_frame=reset_flag||!p.valid?0:p.frame+1;
    bool cold=reset_flag||!p.valid||s.temporal_strength==0.f||!motion.data;
    prepare_k<<<blocks(p.current),128,0,stream>>>(v(work),optional(motion),optional(conf),v(p.current),v(p.mv),v(p.gate),cold,s.temporal_strength,1.f/(mw?mw:std::max(1u,motion.width)),1.f/(mh?mh:std::max(1u,motion.height)));
    Scalars scalars{s.tone,s.structure,-1.f,-1.f,float(s.style)/128.f};
    if(s.automatic_mask){scalars.skin=s.skin>=0.f?s.skin:s.structure;scalars.automatic=s.structure;if(scalars.skin>=0.f&&scalars.automatic>=0.f){scalars.structure=1.f;scalars.skin=-1.f;scalars.automatic=-1.f;}}
    packet_k<<<blocks(p.current),128,0,stream>>>(v(p.current),v(cold?p.current:p.old),v(p.mv),v(p.gate),v(p.sampled),v(p.warped),v(p.packet),p.pending_frame,scalars);check(cudaGetLastError());p.prepared=true;
}
D5Tensor PipelineState::packet()const{return p_->packet;}
D5Tensor PipelineState::current()const{return p_->current;}
D5Tensor PipelineState::sampled_current()const{return p_->sampled;}
D5Tensor PipelineState::warped_history()const{return p_->warped;}
D5Tensor PipelineState::gate()const{return p_->gate;}
D5Tensor PipelineState::history()const{return p_->valid?p_->old:D5Tensor{};}
void PipelineState::composite(const D5Tensor& head,CUstream stream) {auto& p=*p_;require(p.prepared&&stream==p.stream,"prepare and composite require same ordered stream");validate(head,4);same(head,p.current);p.composited=false;composite_k<<<blocks(p.current),128,0,stream>>>(v(p.current),v(p.sampled),v(p.warped),v(p.gate),v(head),v(p.result),v(p.next),p.settings.intensity);check(cudaGetLastError());p.composited=true;}
D5Tensor PipelineState::padded_output()const{return p_->result;}
D5Tensor PipelineState::output()const{D5Tensor t=p_->result;t.width=p_->w;t.height=p_->h;t.row_stride_bytes=uint64_t(p_->nw)*3*4;return t;}
void PipelineState::commit(){auto& p=*p_;require(p.composited,"cannot commit without composite");std::swap(p.old,p.next);p.frame=p.pending_frame;p.valid=true;p.prepared=false;p.composited=false;}
uint32_t PipelineState::neural_width()const{return p_->nw;}
uint32_t PipelineState::neural_height()const{return p_->nh;}
namespace {
__global__ void guide_inputs_k(View color,View motion,View depth,View inputs,View next) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=color.w*color.h)return;int x=i%color.w,y=i/color.w;
    // Match the channel reduction of the pipeline's contiguous BCHW reference.
    float l=(color.get(y,x,0)*.299f+color.get(y,x,1)*.587f)+color.get(y,x,2)*.114f;
    float values[4]={l,depth.get(y,x,0),motion.get(y,x,0)/float(color.w),motion.get(y,x,1)/float(color.h)};
    for(int k=0;k<4;k++){inputs.put(y,x,k,values[k]);next.put(y,x,k,rn(values[k]));}
}
__device__ float guide_sample(View a,float u,float vv,int k,bool linear) {
    if(linear)return texture(a,u,vv,k,true);
    float gx=u*2.f-1.f,gy=vv*2.f-1.f;
    float x=__fmaf_rn(gx+1.f,float(a.w),-1.f)*.5f,y=__fmaf_rn(gy+1.f,float(a.h),-1.f)*.5f;
    x=fminf(fmaxf(x,0.f),float(a.w-1));y=fminf(fmaxf(y,0.f),float(a.h-1));
    return a.get(__float2int_rn(y),__float2int_rn(x),k);
}
__device__ void patch_error(View current,View previous,float u,float vv,float pu,float pv,float& error,float& contrast) {
    float c[9],p[9],cm=0.f,pm=0.f;int n=0;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
        float ox=float(dx)/current.w,oy=float(dy)/current.h;
        c[n]=guide_sample(current,u+ox,vv+oy,0,false);p[n]=guide_sample(previous,pu+ox,pv+oy,0,true);
        cm+=c[n];pm+=p[n];n++;
    }
    cm*=1.f/9.f;pm*=1.f/9.f;error=0.f;contrast=0.f;
    for(int k=0;k<9;k++){float centered=c[k]-cm;error+=fabsf(centered-(p[k]-pm));contrast+=fabsf(centered);}
    error*=1.f/9.f;contrast*=1.f/9.f;
}
__global__ void guide_k(View input,View previous,View provider,View output,View distrust,View tests,bool cold,GuideOptions c) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=input.w*input.h)return;int x=i%input.w,y=i/input.w;
    float u=(float(x)+.5f)*(1.f/input.w),vv=(float(y)+.5f)*(1.f/input.h);
    float fx=input.get(y,x,2),fy=input.get(y,x,3),pu=u+fx,pv=vv+fy,z=input.get(y,x,1);
    float px=provider.get(y,x,0),py=provider.get(y,x,1),length=sqrtf(px*px+py*py);
    float bad[4]={0,0,0,0};
    if(c.static_test){float es,cs,ef,cf;patch_error(input,previous,u,vv,u,vv,es,cs);patch_error(input,previous,u,vv,pu,pv,ef,cf);bad[3]=float(es+.25f*cs<=ef*(1.f+c.static_bias) && cs>=c.min_contrast && length>.5f);}
    if(c.luma){float lo=INFINITY,hi=-INFINITY;for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){float value=guide_sample(input,u+float(dx)/input.w,vv+float(dy)/input.h,0,false);lo=fminf(lo,value);hi=fmaxf(hi,value);}float prev=guide_sample(previous,pu,pv,0,true),margin=c.luma_tolerance*fmaxf(hi,.05f)+2.f/255.f;bad[0]=sat(fmaxf(lo-prev,prev-hi)/margin);}
    if(c.depth){float dp=guide_sample(previous,pu,pv,1,false),tol=c.depth_tolerance*fmaxf(z,.001f);bad[1]=sat((fabsf(dp-z)-tol)/(tol+1e-5f))*float(z<.999f);}
    if(c.consistency && c.mv_consistency>0.f){float dx=(fx-guide_sample(previous,pu,pv,2,false))*input.w,dy=(fy-guide_sample(previous,pu,pv,3,false))*input.h;float diff=sqrtf(dx*dx+dy*dy),allow=c.mv_consistency+.5f*length;bad[2]=sat((diff-allow)/allow);}
    if(!(pu>=0.f && pu<=1.f && pv>=0.f && pv<=1.f)){bad[0]=1.f;bad[1]=bad[2]=bad[3]=0.f;}
    float damp=c.validate?1.f-fmaxf(fmaxf(bad[1],bad[2]),bad[3]):1.f;
    float mask=c.validate?fmaxf(fmaxf(bad[0],bad[1]),bad[2]):0.f;
    float ox=fx*damp,oy=fy*damp;if(cold){ox=fx;oy=fy;mask=1.f;}
    output.put(y,x,0,rn((ox*input.w)*(c.sign_x*c.mv_scale)));output.put(y,x,1,rn((oy*input.h)*(c.sign_y*c.mv_scale)));
    distrust.put(y,x,0,float(__float2int_rn(sat(mask*c.mask_strength)*255.f))*(1.f/255.f));
    for(int k=0;k<4;k++)tests.put(y,x,k,bad[k]);
}
}
struct GuideState::Impl {
    unsigned w=0,h=0;float* data=nullptr;bool valid=false,prepared=false;
    D5Tensor input{},old{},next{},motion{},mask{},tests{};
    ~Impl(){if(data)cudaFree(data);}
};
GuideState::GuideState():p_(new Impl){}
GuideState::~GuideState()=default;
void GuideState::allocate(uint32_t w,uint32_t h){require(w&&h&&uint64_t(w)*h<0x7fffffffULL,"invalid guide extent");if(w==p_->w&&h==p_->h)return;auto q=std::make_unique<Impl>();q->w=w;q->h=h;size_t n=size_t(w)*h;check(cudaMalloc(&q->data,n*19*4));float* d=q->data;auto take=[&](unsigned c){auto t=tensor(d,w,h,c);d+=n*c;return t;};q->input=take(4);q->old=take(4);q->next=take(4);q->motion=take(2);q->mask=take(1);q->tests=take(4);p_=std::move(q);}
void GuideState::reset()noexcept{p_->valid=false;p_->prepared=false;}
void GuideState::prepare(const D5Tensor& color,const D5Tensor& motion,const D5Tensor& depth,bool reset_flag,CUstream stream,const GuideOptions& c){
    auto& p=*p_;validate(color,3);validate(motion,2);validate(depth,1);same(color,motion);same(color,depth);require(p.data&&color.width==p.w&&color.height==p.h,"allocate guide state before prepare");
    if(c.geometry||c.geometry_diagnostics)throw std::runtime_error("native guide geometry fitting is unsupported");
    const float values[]={c.static_bias,c.min_contrast,c.luma_tolerance,c.depth_tolerance,c.mv_consistency,c.mask_strength,c.sign_x,c.sign_y,c.mv_scale};for(float value:values)require(std::isfinite(value),"nonfinite guide setting");require(c.luma_tolerance>=0 && c.depth_tolerance>=0,"negative guide tolerance");
    p.prepared=false;bool cold=reset_flag||!p.valid;
    guide_inputs_k<<<blocks(color),128,0,stream>>>(v(color),v(motion),v(depth),v(p.input),v(p.next));
    guide_k<<<blocks(color),128,0,stream>>>(v(p.input),v(cold?p.input:p.old),v(motion),v(p.motion),v(p.mask),v(p.tests),cold,c);check(cudaGetLastError());p.prepared=true;
}
D5Tensor GuideState::motion()const{return p_->motion;}
D5Tensor GuideState::distrust()const{return p_->mask;}
D5Tensor GuideState::tests()const{return p_->tests;}
void GuideState::commit(){require(p_->prepared,"cannot commit unprepared guide state");std::swap(p_->old,p_->next);p_->valid=true;p_->prepared=false;}
} // namespace dlss5::pipeline
