#define D5_BUILD
#include "dlss5.h"
#include "nr_engine.h"
#include "pipeline.h"
#include "fsr2.h"
#include "frame_ops.h"
#ifdef D5_WITH_GUIDES
#include "guides.h"
#endif
#include <cuda.h>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
thread_local std::string last_error;
struct Failure:std::runtime_error {D5Status status;Failure(D5Status s,const std::string& text):runtime_error(text),status(s){}};
void require(bool b,const char* s){if(!b)throw Failure(D5_INVALID_ARGUMENT,s);}
void cuok(CUresult code){if(code!=CUDA_SUCCESS){const char* name=nullptr;const char* desc=nullptr;cuGetErrorName(code,&name);cuGetErrorString(code,&desc);throw Failure(D5_CUDA_ERROR,std::string(name?name:"CUDA failure")+": "+(desc?desc:""));}}
struct Current {bool pushed=false;explicit Current(CUcontext ctx){cuok(cuCtxPushCurrent(ctx));pushed=true;}~Current(){if(pushed){CUcontext c;cuCtxPopCurrent(&c);}}};
struct DeviceImage {
 CUdeviceptr ptr=0;size_t bytes=0;D5Tensor view{};
 ~DeviceImage(){if(ptr)cuMemFree(ptr);}
 void alloc(uint32_t w,uint32_t h,uint32_t c,uint32_t layout=D5_HWC){size_t n=size_t(w)*h*c*4;if(n!=bytes){if(ptr)cuok(cuMemFree(ptr));ptr=0;bytes=0;cuok(cuMemAlloc(&ptr,n));bytes=n;}view={sizeof(D5Tensor),D5_F32,layout,c,w,h,ptr,layout==D5_HWC?uint64_t(w)*c*4:uint64_t(w)*4,layout==D5_CHW?uint64_t(w)*h*4:0};}
};
uint64_t row(const D5Tensor& t){return t.row_stride_bytes?t.row_stride_bytes:uint64_t(t.width)*4*(t.layout==D5_HWC?t.channels:1);}
void tensor(const D5Tensor& t,uint32_t channels=0,uint32_t layout=D5_HWC){require(t.struct_size>=sizeof(D5Tensor),"Invalid tensor struct_size");require(t.data&&t.width&&t.height,"Empty device tensor");require(t.width<=16384&&t.height<=16384,"Tensor axes exceed 16384");require(t.dtype==D5_F32&&t.layout==layout,"This API path requires Float32 with the documented layout");require(t.channels>0&&t.channels<=16&&(!channels||t.channels==channels),"Unexpected channel count");require(row(t)>=uint64_t(t.width)*4*(layout==D5_HWC?t.channels:1)&&row(t)%4==0,"Invalid row stride");if(layout==D5_CHW)require(!t.plane_stride_bytes||t.plane_stride_bytes>=row(t)*t.height,"Invalid plane stride");}
void device_tensor(const D5Tensor&t,CUcontext context){if(!t.data)return;CUmemorytype type{};auto r=cuPointerGetAttribute(&type,CU_POINTER_ATTRIBUTE_MEMORY_TYPE,t.data);require(r==CUDA_SUCCESS&&(type==CU_MEMORYTYPE_DEVICE||type==CU_MEMORYTYPE_UNIFIED),"Expected CUDA device tensor, not a host address");CUcontext owner=nullptr;cuok(cuPointerGetAttribute(&owner,CU_POINTER_ATTRIBUTE_CONTEXT,t.data));require(!owner||owner==context,"Tensor belongs to another CUDA context");}
void same_size(const D5Tensor&a,const D5Tensor&b){require(a.width==b.width&&a.height==b.height,"Tensor dimensions differ");}
void copy_image(const D5Tensor& a,const D5Tensor& b,CUstream stream){tensor(a,b.channels);tensor(b,a.channels);same_size(a,b);CUDA_MEMCPY2D p{};p.srcMemoryType=CU_MEMORYTYPE_DEVICE;p.dstMemoryType=CU_MEMORYTYPE_DEVICE;p.srcDevice=a.data;p.dstDevice=b.data;p.srcPitch=row(a);p.dstPitch=row(b);p.WidthInBytes=size_t(a.width)*a.channels*4;p.Height=a.height;cuok(cuMemcpy2DAsync(&p,stream));}
void settings(const D5NRSettings&s){require(s.struct_size>=sizeof(s),"Invalid NR settings size");require(s.style<=255,"Style must be an unsigned byte");require(std::isfinite(s.structure)&&std::isfinite(s.tone)&&std::isfinite(s.skin),"Nonfinite NR controls");require(std::isfinite(s.temporal_strength)&&s.temporal_strength>=0&&s.temporal_strength<=1,"Temporal strength must be 0..1");require(std::isfinite(s.intensity)&&s.intensity>=0&&s.intensity<=1,"Intensity must be 0..1");}
template<class F>D5Status protect(F&&f){last_error.clear();try{f();return D5_OK;}catch(const Failure&e){last_error=e.what();return e.status;}catch(const std::exception&e){last_error=e.what();return D5_INTERNAL_ERROR;}catch(...){last_error="Unknown native SDK failure";return D5_INTERNAL_ERROR;}}
}
struct D5SessionImpl {
 D5Config config{};std::string model_dir;std::vector<D5NRSettings> passes;
 CUdevice device=0;CUcontext context=nullptr;CUevent done=nullptr;bool pending=false,have_previous=false;
 std::mutex mutex;std::array<D5Plugin,D5_STAGE_COUNT> plugins{};std::array<std::array<D5Tensor,3>,D5_STAGE_COUNT> last_outputs{};
 std::unique_ptr<dlss5::nr::Engine> nr;
 std::unique_ptr<dlss5::reconstruction::Engine> fsr;
 dlss5::pipeline::GuideOptions guide_options;
 dlss5::pipeline::GuideState guide_state;
#ifdef D5_WITH_GUIDES
 std::unique_ptr<dlss5::guides::Engine> estimators;
#endif
 uint32_t iw=0,ih=0,ow=0,oh=0,ww=0,wh=0;
 DeviceImage source_context,context_image,encoded_source,previous_source,reference,work,base,modified,head,depth,motion,confidence,normalized_depth,invalid_depth,guide_motion,guide_distrust,dilated_motion,fsr_input,reactive,combined_reactive,composition,current_alpha,previous_alpha;
 std::vector<std::unique_ptr<dlss5::pipeline::PipelineState>> states;
 void wait(){if(pending){cuok(cuEventSynchronize(done));pending=false;}}
 void order(CUstream stream){if(pending)cuok(cuStreamWaitEvent(stream,done,0));}
 void mark(CUstream stream){cuok(cuEventRecord(done,stream));pending=true;}
 void invalidate(){last_outputs={};have_previous=false;for(auto& state:states)state->reset();guide_state.reset();if(fsr)fsr->reset();
#ifdef D5_WITH_GUIDES
 if(estimators)estimators->reset_history();
#endif
 }
 bool enabled(uint32_t stage)const{return (config.enabled_stages&(1u<<stage))!=0;}
 bool replacement(uint32_t stage)const{return plugins[stage].process!=nullptr;}
 bool invoke(uint32_t stage,std::initializer_list<D5Tensor> inputs,std::initializer_list<D5Tensor> outputs,const D5Frame& frame,uint32_t pass=0){
  auto& p=plugins[stage];if(!p.process)return false;std::vector<D5Tensor> in(inputs),out(outputs);D5StageInvocation inv{};inv.struct_size=sizeof(inv);inv.stage=stage;inv.frame_id=frame.frame_id;inv.reset=frame.reset||!have_previous;inv.pass_index=pass;inv.cuda_stream=frame.cuda_stream;inv.delta_ms=frame.delta_ms;inv.jitter_x=frame.jitter_x;inv.jitter_y=frame.jitter_y;inv.input_count=(uint32_t)in.size();inv.output_count=(uint32_t)out.size();inv.inputs=in.data();inv.outputs=out.data();inv.nr_settings=pass<passes.size()?&passes[pass]:nullptr;
  auto code=p.process(p.user,&inv);if(code!=D5_OK)throw Failure(code,"Replacement stage "+std::to_string(stage)+" failed with code "+std::to_string(code));return true;
 }
 void ensure_nr(){if(!nr)nr=std::make_unique<dlss5::nr::Engine>(std::filesystem::u8path(model_dir),config.device);}
 void prepare(uint32_t w,uint32_t h){
  require(w>=32&&h>=32&&w<=16384&&h<=16384,"Input axes must be 32..16384");
  uint32_t newow=config.output_width?config.output_width:w,newoh=config.output_height?config.output_height:h;
  uint32_t newww=config.nr_width?config.nr_width:newow,newwh=config.nr_height?config.nr_height:newoh;
  if(iw==w&&ih==h&&ow==newow&&oh==newoh&&ww==newww&&wh==newwh){if(enabled(D5_STAGE_NR)&&!replacement(D5_STAGE_NR)){ensure_nr();nr->prepare((int)states[0]->neural_height(),(int)states[0]->neural_width());}return;}
  wait();invalidate();
#ifdef D5_WITH_GUIDES
 estimators.reset();
#endif
 iw=w;ih=h;ow=newow;oh=newoh;ww=newww;wh=newwh;
  source_context.alloc(iw,ih,3);encoded_source.alloc(iw,ih,3);previous_source.alloc(iw,ih,3);context_image.alloc(ow,oh,3);reference.alloc(ow,oh,3);modified.alloc(ow,oh,3);
  if(enabled(D5_STAGE_NR)){work.alloc(ww,wh,3);base.alloc(ww,wh,3);}depth.alloc(iw,ih,1);motion.alloc(iw,ih,2);confidence.alloc(ow,oh,1);normalized_depth.alloc(iw,ih,1);invalid_depth.alloc(iw,ih,1);guide_motion.alloc(iw,ih,2);dilated_motion.alloc(iw,ih,2);guide_distrust.alloc(iw,ih,1);fsr_input.alloc(iw,ih,3);reactive.alloc(iw,ih,1);combined_reactive.alloc(iw,ih,1);composition.alloc(iw,ih,1);current_alpha.alloc(iw,ih,1);previous_alpha.alloc(iw,ih,1);guide_state.allocate(iw,ih);fsr.reset();
  states.clear();if(enabled(D5_STAGE_NR))for(size_t i=0;i<passes.size();++i){auto state=std::make_unique<dlss5::pipeline::PipelineState>();state->allocate(ww,wh);states.push_back(std::move(state));}
  if(enabled(D5_STAGE_NR))head.alloc(states[0]->neural_width(),states[0]->neural_height(),4);
  if(enabled(D5_STAGE_NR)&&!replacement(D5_STAGE_NR)){ensure_nr();nr->prepare((int)head.view.height,(int)head.view.width);}
 }
 void release(){wait();
#ifdef D5_WITH_GUIDES
 estimators.reset();
#endif
 nr.reset();fsr.reset();states.clear();}
};
namespace {std::unique_lock<std::mutex> lock(D5Session s){require(s!=nullptr,"Null session");std::unique_lock<std::mutex> l(s->mutex,std::try_to_lock);if(!l.owns_lock())throw Failure(D5_BUSY,"Session already has an active host call");return l;}}
extern "C" {
uint32_t D5_CALL d5_abi_version(){return D5_ABI_VERSION;}
const char* D5_CALL d5_last_error(){return last_error.c_str();}
void D5_CALL d5_default_nr_settings(D5NRSettings*s){if(!s)return;*s={sizeof(*s),1,2.f,1.f,-1.f,0,1.f,1.f};}
void D5_CALL d5_default_guide_settings(D5GuideSettings*s){if(!s)return;*s={sizeof(*s),1,1,0,1,1,.15f,.012f,.25f,.10f,1.4f,1.f,1.f,1.f,1.f};}
void D5_CALL d5_default_config(D5Config*c){if(!c)return;std::memset(c,0,sizeof(*c));c->struct_size=sizeof(*c);c->abi_version=D5_ABI_VERSION;c->enabled_stages=D5_ENABLE_NR;c->flow_updates=8;c->flow_longest_side=512;c->depth_input_size=518;c->reference_white_nits=203;c->peak_nits=4000;c->camera_fov_y=1.04719755f;c->output_mix=1;c->nr_pass_count=1;c->camera_near_m=.1f;c->camera_far_m=1000.f;}
D5Status D5_CALL d5_create(const D5Config*c,D5Session*out){return protect([&]{require(out,"Null session output");*out=nullptr;require(c&&c->struct_size>=sizeof(*c)&&c->abi_version==D5_ABI_VERSION,"Unsupported config ABI");require(!(c->enabled_stages&~uint32_t(D5_ENABLE_DEPTH|D5_ENABLE_FLOW|D5_ENABLE_GUIDE|D5_ENABLE_RECONSTRUCT|D5_ENABLE_NR)),"Unknown stage enable bits");require(c->nr_pass_count>=1&&c->nr_pass_count<=30,"NR pass count must be 1..30");require((!c->output_width&&!c->output_height)||(c->output_width>=32&&c->output_height>=32&&c->output_width<=16384&&c->output_height<=16384),"Both output axes must be 32..16384 or zero");require((!c->nr_width&&!c->nr_height)||(c->nr_width>=32&&c->nr_height>=32&&c->nr_width<=16384&&c->nr_height<=16384),"Both NR work axes must be 32..16384 or zero");require(c->input_encoding<=D5_COLOR_PQ2020&&c->output_encoding<=D5_COLOR_PQ2020,"Unknown color encoding");require(std::isfinite(c->output_mix)&&c->output_mix>=0&&c->output_mix<=1,"Output mix must be 0..1");require(std::isfinite(c->reference_white_nits)&&c->reference_white_nits>0&&std::isfinite(c->peak_nits)&&c->peak_nits>0,"Invalid HDR brightness");
 auto s=std::make_unique<D5SessionImpl>();s->config=*c;s->model_dir=c->model_directory_utf8?c->model_directory_utf8:"model";s->config.model_directory_utf8=nullptr;s->config.nr_passes=nullptr;s->config.guide_settings=nullptr;
 require(std::isfinite(c->camera_near_m)&&std::isfinite(c->camera_far_m)&&c->camera_near_m>0&&c->camera_far_m>c->camera_near_m,"Invalid camera near/far");
 if(c->guide_settings){const auto&g=*c->guide_settings;require(g.struct_size>=sizeof(g),"Invalid guide settings");auto&o=s->guide_options;o.validate=g.validate!=0;o.static_test=g.static_test!=0;o.luma=g.luma_test!=0;o.depth=g.depth_test!=0;o.consistency=g.consistency_test!=0;o.static_bias=g.static_bias;o.min_contrast=g.min_contrast;o.luma_tolerance=g.luma_tolerance;o.depth_tolerance=g.depth_tolerance;o.mv_consistency=g.mv_consistency;o.mask_strength=g.mask_strength;o.sign_x=g.sign_x;o.sign_y=g.sign_y;o.mv_scale=g.mv_scale;}s->passes.resize(c->nr_pass_count);for(size_t i=0;i<s->passes.size();++i){if(c->nr_passes)s->passes[i]=c->nr_passes[i];else d5_default_nr_settings(&s->passes[i]);settings(s->passes[i]);}
 cuok(cuInit(0));cuok(cuDeviceGet(&s->device,c->device));int major,minor;cuok(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,s->device));cuok(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,s->device));if(major!=8||minor!=9)throw Failure(D5_UNAVAILABLE,"Current CUDA NR kernels are qualified only for SM89");cuok(cuDevicePrimaryCtxRetain(&s->context,s->device));try{Current current(s->context);cuok(cuEventCreate(&s->done,CU_EVENT_DISABLE_TIMING));}catch(...){cuDevicePrimaryCtxRelease(s->device);throw;}*out=s.release();});}
D5Status D5_CALL d5_capabilities(D5Session s,D5Capabilities*out){return protect([&]{auto l=lock(s);require(out&&out->struct_size>=sizeof(*out),"Invalid capabilities struct");D5Capabilities v{};v.struct_size=sizeof(v);v.abi_version=D5_ABI_VERSION;v.built_in_stages=D5_ENABLE_GUIDE|D5_ENABLE_RECONSTRUCT|(1u<<D5_STAGE_PACKET)|(1u<<D5_STAGE_NR)|(1u<<D5_STAGE_OUTPUT);
#ifdef D5_WITH_GUIDES
 v.built_in_stages|=D5_ENABLE_DEPTH|D5_ENABLE_FLOW;
#endif
 v.replacement_stages=(1u<<D5_STAGE_COUNT)-1;v.compute_major=8;v.compute_minor=9;v.asynchronous=0;*out=v;});}
D5Status D5_CALL d5_set_plugin(D5Session s,const D5Plugin*p){return protect([&]{auto l=lock(s);require(p&&p->struct_size>=sizeof(*p)&&p->stage<D5_STAGE_COUNT,"Invalid replacement stage");Current current(s->context);s->wait();s->plugins[p->stage]=*p;s->invalidate();});}
D5Status D5_CALL d5_set_nr_settings(D5Session s,const D5NRSettings*params,uint32_t count){return protect([&]{auto l=lock(s);require(params&&count>=1&&count<=30,"Expected 1..30 NR settings");for(uint32_t i=0;i<count;++i)settings(params[i]);std::vector<D5NRSettings> next_params(params,params+count);size_t prefix=0;auto equal=[](const D5NRSettings&a,const D5NRSettings&b){return a.style==b.style&&a.structure==b.structure&&a.tone==b.tone&&a.skin==b.skin&&a.automatic_mask==b.automatic_mask&&a.temporal_strength==b.temporal_strength&&a.intensity==b.intensity;};while(prefix<std::min(size_t(count),s->passes.size())&&equal(params[prefix],s->passes[prefix]))++prefix;if(prefix==count&&count==s->passes.size())return;Current current(s->context);s->wait();s->last_outputs={};if(s->iw&&s->enabled(D5_STAGE_NR)){std::vector<std::unique_ptr<dlss5::pipeline::PipelineState>> next(count);for(size_t i=s->states.size();i<count;++i){next[i]=std::make_unique<dlss5::pipeline::PipelineState>();next[i]->allocate(s->ww,s->wh);}for(size_t i=0;i<std::min(s->states.size(),size_t(count));++i){next[i]=std::move(s->states[i]);if(i>=prefix)next[i]->reset();}s->states=std::move(next);}s->passes=std::move(next_params);s->config.nr_pass_count=count;});}
D5Status D5_CALL d5_set_output_mix(D5Session s,float mix){return protect([&]{auto l=lock(s);require(std::isfinite(mix)&&mix>=0&&mix<=1,"Output mix must be 0..1");s->config.output_mix=mix;});}
D5Status D5_CALL d5_prepare(D5Session s,uint32_t w,uint32_t h){return protect([&]{auto l=lock(s);Current current(s->context);s->prepare(w,h);});}
D5Status D5_CALL d5_process(D5Session s,const D5Frame*f){return protect([&]{auto l=lock(s);require(f&&f->struct_size>=sizeof(*f),"Invalid frame struct");tensor(f->color);require(f->color.channels==3||f->color.channels==4,"Color must be HWC RGB or RGBA");tensor(f->output);require(f->output.channels==3||f->output.channels==4,"Output must be HWC RGB or RGBA");require(std::isfinite(f->jitter_x)&&std::isfinite(f->jitter_y)&&std::isfinite(f->delta_ms)&&f->delta_ms>0,"Invalid frame timing/jitter");require(s->enabled(D5_STAGE_RECONSTRUCT)||(f->jitter_x==0&&f->jitter_y==0),"Nonzero jitter requires reconstruction stage");
 require(!(s->enabled(D5_STAGE_RECONSTRUCT)&&!s->replacement(D5_STAGE_RECONSTRUCT)&&s->config.input_encoding>=D5_COLOR_LINEAR709_NITS),"Built-in FSR currently supports SDR only; HDR requires an HDR reconstruction replacement or disabled reconstruction");Current current(s->context);for(const auto* t:{&f->color,&f->depth,&f->motion,&f->confidence,&f->reactive,&f->composition,&f->protect,&f->output})device_tensor(*t,s->context);s->prepare(f->color.width,f->color.height);require(f->output.width==s->ow&&f->output.height==s->oh,"Output buffer does not match configured dimensions");auto stream=(CUstream)f->cuda_stream;s->order(stream);
 using dlss5::pipeline::Ops;dlss5::pipeline::ColorOptions colors{s->config.input_encoding,s->config.output_encoding,s->config.reference_white_nits,s->config.peak_nits};
 try{
 if(f->reset)s->invalidate();Ops::preprocess(f->color,s->source_context.view,colors,stream);Ops::encode(s->source_context.view,s->encoded_source.view,colors,stream);dlss5::frame_ops::alpha(f->color,s->encoded_source.view,s->current_alpha.view,stream);
 D5Tensor depth=f->depth,motion=f->motion,confidence=f->confidence;if(depth.data)tensor(depth,1);if(motion.data)tensor(motion,2);if(confidence.data)tensor(confidence,1);if(f->protect.data)tensor(f->protect,1);
#ifdef D5_WITH_GUIDES
 if(!s->estimators&&((s->enabled(D5_STAGE_DEPTH)&&!depth.data&&!s->replacement(D5_STAGE_DEPTH))||(s->enabled(D5_STAGE_FLOW)&&!motion.data&&!s->replacement(D5_STAGE_FLOW)))){dlss5::guides::Config gc;gc.depth=s->enabled(D5_STAGE_DEPTH);gc.flow=s->enabled(D5_STAGE_FLOW);gc.flow_updates=(int)s->config.flow_updates;gc.flow_longest_side=(int)s->config.flow_longest_side;gc.depth_input_size=(int)s->config.depth_input_size;s->estimators=std::make_unique<dlss5::guides::Engine>((std::filesystem::u8path(s->model_dir)/"native_guides").u8string(),s->config.device,gc);}
#endif
 if(s->enabled(D5_STAGE_DEPTH)&&!depth.data){depth=s->depth.view;if(!s->invoke(D5_STAGE_DEPTH,{s->encoded_source.view},{depth},*f)){
#ifdef D5_WITH_GUIDES
 s->estimators->estimate_depth((float*)s->encoded_source.ptr,s->ih,s->iw,(float*)depth.data,stream);
#else
 throw Failure(D5_UNAVAILABLE,"VDA backend was not included in this build");
#endif
 }}
 if(s->enabled(D5_STAGE_FLOW)&&!motion.data){motion=s->motion.view;const auto& prev=s->have_previous?s->previous_source.view:s->encoded_source.view;if(!s->invoke(D5_STAGE_FLOW,{s->encoded_source.view,prev},{motion},*f)){
#ifdef D5_WITH_GUIDES
 s->estimators->estimate_flow((float*)s->encoded_source.ptr,(float*)prev.data,s->ih,s->iw,(float*)motion.data,stream);
#else
 throw Failure(D5_UNAVAILABLE,"RAFT backend was not included in this build");
#endif
 }}

 D5Tensor raw_motion=motion,guide_motion_view{},distrust{};

 D5Tensor normalized{},invalid{};
 if(s->enabled(D5_STAGE_GUIDE)||s->enabled(D5_STAGE_RECONSTRUCT)){
  bool builtin=(s->enabled(D5_STAGE_GUIDE)&&!s->replacement(D5_STAGE_GUIDE))||(s->enabled(D5_STAGE_RECONSTRUCT)&&!s->replacement(D5_STAGE_RECONSTRUCT));
  if(builtin)require(depth.data&&motion.data,"Built-in guide/reconstruction requires supplied or estimated depth and motion");
  if(depth.data){require(depth.width==s->iw&&depth.height==s->ih,"Depth must be on source grid for guide/reconstruction");if(depth.data!=s->depth.ptr){copy_image(depth,s->depth.view,stream);depth=s->depth.view;}dlss5::reconstruction::normalize_depth((float*)depth.data,(float*)s->normalized_depth.ptr,(float*)s->invalid_depth.ptr,s->iw,s->ih,s->config.camera_near_m,s->config.camera_far_m,stream);normalized=s->normalized_depth.view;invalid=s->invalid_depth.view;}
  if(motion.data){require(motion.width==s->iw&&motion.height==s->ih,"Motion must be on source grid for guide/reconstruction");if(motion.data!=s->motion.ptr){copy_image(motion,s->motion.view,stream);motion=s->motion.view;}}
 }

 if(s->enabled(D5_STAGE_GUIDE)){
  if(s->invoke(D5_STAGE_GUIDE,{s->encoded_source.view,depth,motion,normalized},{s->guide_motion.view,s->guide_distrust.view},*f)){motion=s->guide_motion.view;distrust=s->guide_distrust.view;}
  else{s->guide_state.prepare(s->encoded_source.view,motion,s->normalized_depth.view,f->reset!=0,stream,s->guide_options);motion=s->guide_state.motion();distrust=s->guide_state.distrust();}
 }
 guide_motion_view=motion;
 if(s->enabled(D5_STAGE_RECONSTRUCT)){
  D5Tensor user_reactive=f->reactive,user_composition=f->composition;
  if(user_reactive.data){tensor(user_reactive,1);copy_image(user_reactive,s->reactive.view,stream);user_reactive=s->reactive.view;}
  if(user_composition.data){tensor(user_composition,1);copy_image(user_composition,s->composition.view,stream);user_composition=s->composition.view;}
  float white=s->config.input_encoding>=D5_COLOR_LINEAR709_NITS?s->config.reference_white_nits:1.f;
  dlss5::frame_ops::fsr_input(s->source_context.view,s->current_alpha.view,s->previous_alpha.view,!s->have_previous,invalid,user_reactive,s->fsr_input.view,s->combined_reactive.view,white,stream);
  if(s->invoke(D5_STAGE_RECONSTRUCT,{s->fsr_input.view,depth,motion,s->combined_reactive.view,user_composition,distrust},{s->context_image.view,s->confidence.view,s->dilated_motion.view},*f)){motion=s->dilated_motion.view;}
  else{
   if(!s->fsr){dlss5::reconstruction::Settings settings;settings.output_width=s->ow;settings.output_height=s->oh;settings.camera_near=s->config.camera_near_m;settings.camera_far=s->config.camera_far_m;settings.camera_fov_y=s->config.camera_fov_y;settings.encoding=1;s->fsr=std::make_unique<dlss5::reconstruction::Engine>(s->iw,s->ih,settings);}
   s->fsr->process((float*)s->fsr_input.ptr,(float*)depth.data,(float*)motion.data,(float*)s->combined_reactive.ptr,(float*)user_composition.data,(float*)s->context_image.ptr,(float*)s->confidence.ptr,stream,f->jitter_x,f->jitter_y,f->delta_ms);
   motion=s->motion.view;motion.data=(uint64_t)s->fsr->dilated_motion();
  }
  dlss5::frame_ops::fsr_restore(s->source_context.view,s->context_image.view,white,stream);
  if(!confidence.data)confidence=s->confidence.view;
 }else{
  require(!f->reactive.data&&!f->composition.data,"Reactive/composition masks require reconstruction enabled");Ops::resize(s->source_context.view,s->context_image.view,stream);
 }

 Ops::encode(s->context_image.view,s->reference.view,colors,stream);Ops::half_rtz(s->reference.view,s->reference.view,stream);
 if(s->enabled(D5_STAGE_NR)){
 Ops::resize(s->reference.view,s->work.view,stream);Ops::half_rtz(s->work.view,s->base.view,stream);D5Tensor work=s->base.view;
 for(uint32_t p=0;p<s->passes.size();++p){auto& state=*s->states[p];state.prepare(work,motion,confidence,s->passes[p],f->reset!=0,stream,s->iw,s->ih);auto packet=state.packet();s->invoke(D5_STAGE_PACKET,{state.sampled_current(),state.warped_history(),state.gate()},{packet},*f,p);if(!s->invoke(D5_STAGE_NR,{packet},{s->head.view},*f,p)){s->ensure_nr();s->nr->infer(packet.data,s->head.ptr,packet.height,packet.width,stream);}state.composite(s->head.view,stream);work=state.output();}
 if(s->ww==s->ow&&s->wh==s->oh)copy_image(work,s->modified.view,stream);else Ops::transport(s->reference.view,work,s->base.view,s->modified.view,stream);
 }else copy_image(s->reference.view,s->modified.view,stream);
 if(!s->invoke(D5_STAGE_OUTPUT,{s->context_image.view,s->reference.view,s->modified.view,f->color,f->protect},{f->output},*f))Ops::finish(s->context_image.view,s->reference.view,s->modified.view,f->color,f->protect,f->output,colors,s->config.output_mix,stream);
 s->last_outputs={};if(depth.data)s->last_outputs[D5_STAGE_DEPTH][0]=depth;if(raw_motion.data)s->last_outputs[D5_STAGE_FLOW][0]=raw_motion;
 if(s->enabled(D5_STAGE_GUIDE)){s->last_outputs[D5_STAGE_GUIDE][0]=guide_motion_view;s->last_outputs[D5_STAGE_GUIDE][1]=distrust;s->last_outputs[D5_STAGE_GUIDE][2]=normalized;}
 if(s->enabled(D5_STAGE_RECONSTRUCT)){s->last_outputs[D5_STAGE_RECONSTRUCT][0]=s->context_image.view;s->last_outputs[D5_STAGE_RECONSTRUCT][1]=confidence;s->last_outputs[D5_STAGE_RECONSTRUCT][2]=motion;}
 if(s->enabled(D5_STAGE_NR)){s->last_outputs[D5_STAGE_PACKET][0]=s->states.back()->packet();s->last_outputs[D5_STAGE_NR][0]=s->head.view;}s->last_outputs[D5_STAGE_OUTPUT][0]=f->output;
 copy_image(s->encoded_source.view,s->previous_source.view,stream);copy_image(s->current_alpha.view,s->previous_alpha.view,stream);s->mark(stream);if(s->enabled(D5_STAGE_GUIDE)&&!s->replacement(D5_STAGE_GUIDE))s->guide_state.commit();if(s->enabled(D5_STAGE_NR))for(auto& state:s->states)state->commit();s->have_previous=true;
 }catch(...){cuStreamSynchronize(stream);s->invalidate();throw;}
 });}
D5Status D5_CALL d5_prepare_packet(D5Session s,uint32_t w,uint32_t h){return protect([&]{auto l=lock(s);require(w>=64&&h>=64&&w%64==0&&h%64==0&&w<=16384&&h<=16384,"Packet axes must be 64-aligned and 64..16384");Current current(s->context);s->wait();s->ensure_nr();s->nr->prepare(h,w);});}
D5Status D5_CALL d5_infer_packet(D5Session s,const D5Tensor*packet,const D5Tensor*head,void*stream){return protect([&]{auto l=lock(s);require(packet&&head,"Null packet/head");tensor(*packet,16,D5_CHW);tensor(*head,4);same_size(*packet,*head);require(row(*packet)==uint64_t(packet->width)*4&&(!packet->plane_stride_bytes||packet->plane_stride_bytes==uint64_t(packet->width)*packet->height*4)&&row(*head)==uint64_t(head->width)*16,"Direct NR requires contiguous packet/head");Current current(s->context);s->order((CUstream)stream);s->ensure_nr();try{s->nr->infer(packet->data,head->data,packet->height,packet->width,(CUstream)stream);s->last_outputs={};s->last_outputs[D5_STAGE_PACKET][0]=*packet;s->last_outputs[D5_STAGE_NR][0]=*head;s->mark((CUstream)stream);}catch(...){cuStreamSynchronize((CUstream)stream);throw;}});}
D5Status D5_CALL d5_stage_output(D5Session s,uint32_t stage,uint32_t index,D5Tensor*out){return protect([&]{auto l=lock(s);require(out&&out->struct_size>=sizeof(*out)&&stage<D5_STAGE_COUNT&&index<3,"Invalid stage output query");const auto& view=s->last_outputs[stage][index];if(!view.data)throw Failure(D5_UNAVAILABLE,"Stage output is not available for this frame");*out=view;});}
D5Status D5_CALL d5_synchronize(D5Session s){return protect([&]{auto l=lock(s);Current current(s->context);s->wait();});}
D5Status D5_CALL d5_reset(D5Session s){return protect([&]{auto l=lock(s);Current current(s->context);s->wait();s->invalidate();
#ifdef D5_WITH_GUIDES
 if(s->estimators)s->estimators->reset();
#endif
 });}
D5Status D5_CALL d5_destroy(D5Session s){if(!s)return D5_OK;return protect([&]{auto l=lock(s);CUdevice dev=s->device;CUcontext context=s->context;{Current current(context);s->release();if(s->done)cuok(cuEventDestroy(s->done));l.unlock();delete s;}cuok(cuDevicePrimaryCtxRelease(dev));});}
}
