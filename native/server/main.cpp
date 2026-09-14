#include "dlss5.h"
#include "image.h"
#include "httplib.h"
#include "json.hpp"
#include <cuda.h>
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <atomic>
#include <chrono>
#include <array>

using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace {
void check(D5Status s){if(s)throw std::runtime_error(std::string(d5_last_error()));}
void cuda(CUresult r){if(r){const char*t=nullptr;cuGetErrorString(r,&t);throw std::runtime_error(t?t:"CUDA error");}}
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::string id(){static std::atomic<uint64_t> next{1};std::ostringstream s;s<<std::hex<<std::chrono::system_clock::now().time_since_epoch().count()<<next++;return s.str();}
struct Mem{CUdeviceptr p=0;explicit Mem(size_t bytes){cuda(cuMemAlloc(&p,bytes));}~Mem(){if(p)cuMemFree(p);}};
struct Stream{CUstream p=nullptr;Stream(){cuda(cuStreamCreate(&p,CU_STREAM_NON_BLOCKING));}~Stream(){if(p){cuStreamSynchronize(p);cuStreamDestroy(p);}}};
D5Tensor image_view(CUdeviceptr p,uint32_t w,uint32_t h,uint32_t channels=4){return {sizeof(D5Tensor),D5_F32,D5_HWC,channels,w,h,p,uint64_t(w)*channels*4,0};}
struct Session {
 D5Session sdk=nullptr;std::string config_key;uint64_t frame=0;CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;
 explicit Session(int index){cuda(cuInit(0));cuda(cuDeviceGet(&device,index));cuda(cuDevicePrimaryCtxRetain(&context,device));cuda(cuCtxPushCurrent(context));auto r=cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING);CUcontext old;cuCtxPopCurrent(&old);if(r){cuDevicePrimaryCtxRelease(device);cuda(r);}}
 ~Session(){if(context){cuCtxPushCurrent(context);if(sdk)d5_destroy(sdk);if(stream){cuStreamSynchronize(stream);cuStreamDestroy(stream);}CUcontext old;cuCtxPopCurrent(&old);cuDevicePrimaryCtxRelease(device);}}
};
struct Job {std::string id,status="queued",error,upload,png,stream_id;json config,metrics;double created_at=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();};
class Jobs {
 std::mutex mutex;std::condition_variable cv;std::deque<std::shared_ptr<Job>> queue;std::map<std::string,std::shared_ptr<Job>> records;std::deque<std::string> order;std::map<std::string,std::unique_ptr<Session>> sessions;bool stop=false;std::thread worker;std::string model_dir;int device;
 void status(const std::shared_ptr<Job>&j,const char*s){std::lock_guard<std::mutex> lock(mutex);j->status=s;}
 void loop(){for(;;){std::shared_ptr<Job> j;{std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return stop||!queue.empty();});if(stop&&queue.empty())break;j=queue.front();queue.pop_front();}auto start=Clock::now();try{status(j,"preparing");auto input=d5server::decode_image(j->upload);j->upload.clear();auto cfg=j->config;
  D5Config c;d5_default_config(&c);c.model_directory_utf8=model_dir.c_str();c.device=device;auto value=[&](const char*key,uint32_t fallback){return cfg.value(key,fallback);};c.output_width=value("width",0);c.output_height=value("height",0);if(!c.output_width)c.output_width=input.width;if(!c.output_height)c.output_height=input.height;c.nr_width=value("nr_width",0);c.nr_height=value("nr_height",0);c.output_mix=cfg.value("mix",1.f);c.flow_updates=value("flow_updates",8);c.flow_longest_side=value("flow_longest_side",512);c.depth_input_size=value("depth_input_size",518);
  c.enabled_stages=cfg.value("nr",true)?D5_ENABLE_NR:0;if(cfg.value("depth",false))c.enabled_stages|=D5_ENABLE_DEPTH;if(cfg.value("flow",false))c.enabled_stages|=D5_ENABLE_FLOW;if(cfg.value("guide",false))c.enabled_stages|=D5_ENABLE_GUIDE;if(cfg.value("reconstruction",false))c.enabled_stages|=D5_ENABLE_RECONSTRUCT;
  c.nr_pass_count=value("passes",1);if(c.nr_pass_count>30||c.nr_pass_count<1)throw std::runtime_error("passes must be 1..30");std::vector<D5NRSettings> passes(c.nr_pass_count);for(auto&p:passes){d5_default_nr_settings(&p);p.structure=cfg.value("structure",2.f);p.tone=cfg.value("tone",1.f);p.style=value("style",1);p.skin=cfg.value("skin",-1.f);p.automatic_mask=cfg.value("automatic_mask",false);p.temporal_strength=cfg.value("temporal_strength",1.f);p.intensity=cfg.value("intensity",1.f);}c.nr_passes=passes.data();
  json key=cfg;key.erase("reset");key.erase("stream_id");for(const char* control:{"structure","tone","style","skin","automatic_mask","temporal_strength","intensity","passes","mix","delta_ms"})key.erase(control);key["input_width"]=input.width;key["input_height"]=input.height;std::string sid=j->stream_id.empty()?"__independent":j->stream_id;auto it=sessions.find(sid);if(it==sessions.end()){if(sessions.size()>=4)throw std::runtime_error("At most 4 stream sessions; restart server to release unused sessions");it=sessions.emplace(sid,std::make_unique<Session>(device)).first;}Session&s=*it->second;std::string signature=key.dump();if(!s.sdk||s.config_key!=signature){if(s.sdk){check(d5_destroy(s.sdk));s.sdk=nullptr;}check(d5_create(&c,&s.sdk));s.config_key=signature;s.frame=0;}else{check(d5_set_nr_settings(s.sdk,passes.data(),c.nr_pass_count));check(d5_set_output_mix(s.sdk,c.output_mix));}
  check(d5_prepare(s.sdk,input.width,input.height));double setup_seconds=elapsed(start);status(j,"processing");auto process=Clock::now();CUdevice dev;CUcontext context;cuda(cuDeviceGet(&dev,device));cuda(cuDevicePrimaryCtxRetain(&context,dev));struct Context {CUdevice dev;Context(CUcontext c,CUdevice d):dev(d){cuda(cuCtxPushCurrent(c));}~Context(){CUcontext c;cuCtxPopCurrent(&c);cuDevicePrimaryCtxRelease(dev);}} scoped(context,dev);
  CUstream stream=s.stream;Mem in(input.rgb.size()*4),out(size_t(c.output_width)*c.output_height*16);cuda(cuMemcpyHtoD(in.p,input.rgb.data(),input.rgb.size()*4));D5Frame frame{};frame.struct_size=sizeof(frame);frame.reset=cfg.value("reset",j->stream_id.empty())||s.frame==0;frame.frame_id=s.frame;frame.cuda_stream=stream;frame.color=image_view(in.p,input.width,input.height);frame.output=image_view(out.p,c.output_width,c.output_height);frame.delta_ms=cfg.value("delta_ms",1000.f/60.f);check(d5_process(s.sdk,&frame));d5server::Image result;result.width=c.output_width;result.height=c.output_height;result.rgb.resize(size_t(result.width)*result.height*4);check(d5_synchronize(s.sdk));cuda(cuStreamSynchronize(stream));cuda(cuMemcpyDtoH(result.rgb.data(),out.p,result.rgb.size()*4));++s.frame;double infer_seconds=elapsed(process);auto png=d5server::encode_png(result);
  {std::lock_guard<std::mutex> lock(mutex);j->png=std::move(png);j->metrics={{"setup_seconds",setup_seconds},{"process_seconds_including_upload_download",infer_seconds},{"total_seconds",elapsed(start)},{"input_width",input.width},{"input_height",input.height},{"width",result.width},{"height",result.height},{"frame",s.frame-1},{"enabled_stages",c.enabled_stages}};j->status="completed";}
 }catch(const std::exception&e){std::lock_guard<std::mutex>lock(mutex);j->status="failed";j->error=e.what();j->upload.clear();} }sessions.clear();}
 public:Jobs(std::string dir,int dev):model_dir(std::move(dir)),device(dev){worker=std::thread([this]{loop();});}~Jobs(){{std::lock_guard<std::mutex>lock(mutex);stop=true;}cv.notify_one();worker.join();}
 std::string submit(std::string upload,json cfg){if(!cfg.is_object())throw std::runtime_error("config must be a JSON object");const std::vector<std::string> allowed={"width","height","nr_width","nr_height","mix","flow_updates","flow_longest_side","depth_input_size","nr","depth","flow","guide","reconstruction","passes","structure","tone","style","skin","automatic_mask","temporal_strength","intensity","reset","stream_id","delta_ms"};for(auto it=cfg.begin();it!=cfg.end();++it)if(std::find(allowed.begin(),allowed.end(),it.key())==allowed.end())throw std::runtime_error("Unsupported server setting: "+it.key());std::lock_guard<std::mutex>lock(mutex);if(queue.size()>=4)throw std::runtime_error("Job queue full");auto j=std::make_shared<Job>();j->id=::id();j->upload=std::move(upload);j->config=std::move(cfg);j->stream_id=j->config.value("stream_id",std::string{});if(j->stream_id.size()>128)throw std::runtime_error("stream_id too long");while(order.size()>=8){auto old=records.find(order.front());if(old->second->status!="completed"&&old->second->status!="failed")throw std::runtime_error("Too many unfinished jobs");records.erase(old);order.pop_front();}records[j->id]=j;order.push_back(j->id);queue.push_back(j);cv.notify_one();return j->id;}
 static json snapshot(const Job& j){return {{"id",j.id},{"status",j.status},{"error",j.error},{"stream_id",j.stream_id},{"metrics",j.metrics},{"config",j.config},{"created_at",j.created_at},{"result",j.status=="completed"?"/v1/jobs/"+j.id+"/result.png":""}};}
 json get(const std::string& id){std::lock_guard<std::mutex>lock(mutex);auto it=records.find(id);if(it==records.end())throw std::runtime_error("Job not found");return snapshot(*it->second);}
 json list(){std::lock_guard<std::mutex>lock(mutex);json items=json::array();for(auto it=order.rbegin();it!=order.rend();++it)items.push_back(snapshot(*records.at(*it)));return {{"jobs",items}};}

 std::string result(const std::string&id){std::lock_guard<std::mutex>lock(mutex);auto it=records.find(id);if(it==records.end()||it->second->status!="completed")throw std::runtime_error("Result not available");return it->second->png;}
};
std::filesystem::path exe_dir(){std::wstring path(32768,L'\0');DWORD n=GetModuleFileNameW(nullptr,path.data(),(DWORD)path.size());path.resize(n);return std::filesystem::path(path).parent_path();}
}
void configure_external_runtime(const std::filesystem::path& directory);
int main(int argc,char**argv){try{
 std::string host="127.0.0.1",token;int port=7863,device=0;std::filesystem::path models=exe_dir()/"model",runtime=exe_dir()/"runtime",assets=exe_dir()/"assets";
 for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&](){if(++i>=argc)throw std::runtime_error("Missing option value");return std::string(argv[i]);};if(a=="--port")port=std::stoi(next());else if(a=="--host")host=next();else if(a=="--token")token=next();else if(a=="--model")models=std::filesystem::u8path(next());else if(a=="--runtime")runtime=std::filesystem::u8path(next());else if(a=="--assets")assets=std::filesystem::u8path(next());else if(a=="--device")device=std::stoi(next());else if(a=="--help"){std::cout<<"dlss5_server [--model PATH] [--runtime PATH] [--assets PATH] [--port 7863] [--host 127.0.0.1] [--token SECRET] [--device 0]\n";return 0;}else throw std::runtime_error("Unknown option "+a);}
 if(host!="127.0.0.1"&&host!="localhost"&&host!="::1"&&token.empty())throw std::runtime_error("Non-loopback binding requires --token; use TLS reverse proxy on untrusted networks");models=std::filesystem::absolute(models);runtime=std::filesystem::absolute(runtime);assets=std::filesystem::absolute(assets);configure_external_runtime(runtime);Jobs jobs(models.u8string(),device);httplib::Server server;server.set_payload_max_length(32u*1024u*1024u);server.set_read_timeout(30,0);server.set_write_timeout(30,0);server.new_task_queue=[](){return new httplib::ThreadPool(4);};
 server.set_pre_routing_handler([&](const httplib::Request&r,httplib::Response&s){const bool api=r.path.rfind("/v1/",0)==0||r.path.rfind("/api/",0)==0;if(api&&!token.empty()&&r.get_header_value("Authorization")!="Bearer "+token){s.status=401;s.set_content("{\"error\":\"Bearer token required\"}","application/json");return httplib::Server::HandlerResponse::Handled;}if(api&&r.has_header("Origin")&&r.get_header_value("Origin")!="http://"+r.get_header_value("Host")){s.status=403;s.set_content("{\"error\":\"Cross-origin request rejected\"}","application/json");return httplib::Server::HandlerResponse::Handled;}return httplib::Server::HandlerResponse::Unhandled;});
 if(std::filesystem::is_regular_file(assets/"index.html")){
  if(!server.set_mount_point("/",assets.u8string()))throw std::runtime_error("Cannot mount web assets: "+assets.u8string());
  server.set_file_request_handler([](const auto& r,auto& response){response.set_header("X-Content-Type-Options","nosniff");response.set_header("Cache-Control",r.path=="/"||r.path=="/index.html"?"no-cache":"public, max-age=31536000, immutable");});
 }else{
  std::cerr<<"Web assets missing: "<<assets.u8string()<<" (build native/webui and pass --assets)\n";
  server.Get("/",[](const auto&,auto&response){response.status=503;response.set_content("Web assets missing. Build native/webui with npm run build and set --assets to its dist directory.","text/plain; charset=utf-8");});
 }
 auto info=[&](const auto&,auto&response){json available=json::array();if(std::filesystem::is_regular_file(models/"weights_ht_blob.bin"))available.push_back("weights_ht_blob.bin");if(std::filesystem::is_directory(models/"native_guides"))for(const auto& entry:std::filesystem::directory_iterator(models/"native_guides"))if(entry.is_regular_file()&&entry.path().extension()==".onnx")available.push_back("native_guides/"+entry.path().filename().u8string());
  response.set_content(json({{"abi_version",d5_abi_version()},{"implementation","native CUDA, no Python"},{"model_directory",models.u8string()},{"assets_directory",assets.u8string()},{"runtime_directory",runtime.u8string()},{"assets_present",std::filesystem::is_regular_file(assets/"index.html")},{"runtime_present",std::filesystem::is_regular_file(runtime/"bin/onnxruntime.dll")&&std::filesystem::is_regular_file(runtime/"cuda12.8/nvrtc/bin/nvrtc64_120_0.dll")},{"available_models",available},{"device",device},{"packet_layout","F32 CHW16 -> HWC4"},{"gpu_target","SM89"},{"http_images","RGB/RGBA PNG/JPEG/BMP; first frame; HWC GPU C ABI for advanced inputs"}}).dump(),"application/json");
 };server.Get("/v1/info",info);server.Get("/api/status",info);
 server.Get("/v1/jobs",[&](const auto&,auto&response){response.set_content(jobs.list().dump(),"application/json");});
 auto submit=[&](const httplib::Request&r,httplib::Response&s){try{if(!r.has_file("image"))throw std::runtime_error("multipart image field required");json cfg=json::object();if(r.has_file("config"))cfg=json::parse(r.get_file_value("config").content);auto id=jobs.submit(r.get_file_value("image").content,cfg);s.status=202;s.set_content(json({{"id",id},{"status_url","/v1/jobs/"+id}}).dump(),"application/json");}catch(const std::exception&e){s.status=400;s.set_content(json({{"error",e.what()}}).dump(),"application/json");}};server.Post("/v1/jobs",submit);server.Post("/api/jobs",submit);
 server.Get(R"(/v1/jobs/([a-f0-9]+)/result\.png)",[&](const auto&r,auto&s){try{s.set_content(jobs.result(r.matches[1]),"image/png");}catch(const std::exception&e){s.status=404;s.set_content(json({{"error",e.what()}}).dump(),"application/json");}});
 server.Get(R"(/v1/jobs/([a-f0-9]+))",[&](const auto&r,auto&s){try{s.set_content(jobs.get(r.matches[1]).dump(),"application/json");}catch(const std::exception&e){s.status=404;s.set_content(json({{"error",e.what()}}).dump(),"application/json");}});
 server.Get("/v1/openapi.json",[](const auto&,auto&response){
  json schema;schema["openapi"]="3.0.3";schema["info"]={{"title","DLSS5 Native Server"},{"version","1"}};
  auto param=json::array({{{"name","id"},{"in","path"},{"required",true},{"schema",{{"type","string"}}}}});
  schema["paths"]["/v1/info"]["get"]["responses"]["200"]={{"description","SDK information"}};
  schema["paths"]["/v1/jobs"]["get"]["responses"]["200"]={{"description","Up to eight recent jobs, with effective submitted config and created_at Unix seconds"}};
  auto& post=schema["paths"]["/v1/jobs"]["post"];post["summary"]="Queue native processing";
  post["requestBody"]["required"]=true;
  post["requestBody"]["content"]["multipart/form-data"]["schema"]={{"type","object"},{"required",json::array({"image"})},{"properties",{{"image",{{"type","string"},{"format","binary"}}},{"config",{{"type","string"},{"description","JSON: width,height,structure,tone,style,skin,passes,mix,depth,flow,guide,reset,stream_id"}}}}}};
  post["responses"]["202"]={{"description","Queued job ID"}};
  for(const char* path:{"/v1/jobs/{id}","/v1/jobs/{id}/result.png"}){auto& get=schema["paths"][path]["get"];get["parameters"]=param;get["responses"]["200"]={{"description","Job status or output PNG"}};}
  response.set_content(schema.dump(),"application/json");
 });
 std::cout<<"DLSS5 native server http://"<<host<<":"<<port<<"\nModels: "<<models.u8string()<<"\nRuntime: "<<runtime.u8string()<<"\nWeb assets: "<<assets.u8string()<<"\n";if(!server.listen(host,port))throw std::runtime_error("Cannot listen on requested endpoint");return 0;
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}}
