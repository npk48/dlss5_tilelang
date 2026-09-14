"""CUDA12.2 native pipeline vs current CUDA PyTorch reference; no model needed."""
import ctypes as C
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reference'))
import dlss5_model as nr
from whitebox_pipeline.fsr2 import half as half_rtz
from whitebox_pipeline.color import ColorBridge,ColorSettings,decode_input
from whitebox_pipeline.post import composite,OutputSettings
class Tensor(C.Structure):
    _fields_=[(k,C.c_uint32) for k in ('struct_size','dtype','layout','channels','width','height')]+[(k,C.c_uint64) for k in ('data','row_stride_bytes','plane_stride_bytes')]
class Settings(C.Structure):
    _fields_=[('struct_size',C.c_uint32),('style',C.c_uint32),('structure',C.c_float),('tone',C.c_float),('skin',C.c_float),('automatic_mask',C.c_uint32),('temporal_strength',C.c_float),('intensity',C.c_float)]
class Color(C.Structure):
    _fields_=[('input_encoding',C.c_uint32),('output_encoding',C.c_uint32),('reference_white_nits',C.c_float),('peak_nits',C.c_float)]
def view(a,chw=False):
    if a is None:return Tensor()
    if chw:c,h,w=a.shape;row=a.stride(1)*4;plane=a.stride(0)*4
    else:h,w,c=a.shape;row=a.stride(0)*4;plane=0
    return Tensor(C.sizeof(Tensor),1,2 if chw else 1,c,w,h,a.data_ptr(),row,plane)
def ptr(a):return C.byref(a)
lib=C.CDLL(str(ROOT/'native/pipeline/build/dlss5_pipeline_test.dll'))
lib.pipeline_error.restype=C.c_char_p
lib.pipeline_create.argtypes=[C.c_uint32,C.c_uint32];lib.pipeline_create.restype=C.c_void_p
P=C.POINTER(Tensor)
lib.pipeline_destroy.argtypes=[C.c_void_p]
lib.pipeline_prepare.argtypes=[C.c_void_p,P,P,P,C.POINTER(Settings),C.c_int,C.c_void_p]
lib.pipeline_view.argtypes=[C.c_void_p,C.c_int,P]
lib.pipeline_copy.argtypes=[P,P,C.c_void_p]
lib.pipeline_composite.argtypes=[C.c_void_p,P,C.c_void_p]
lib.pipeline_commit.argtypes=[C.c_void_p]
lib.pipeline_resize.argtypes=[P,P,C.c_int,C.c_void_p]
lib.pipeline_color.argtypes=[P,P,C.POINTER(Color),C.c_int,C.c_void_p]
lib.pipeline_finish.argtypes=[P,P,P,P,P,P,C.POINTER(Color),C.c_float,C.c_void_p]
lib.pipeline_transport.argtypes=[P,P,P,P,C.c_void_p]
lib.pipeline_reconstruct.argtypes=[P,P,C.c_int,C.c_void_p]
stream=torch.cuda.Stream()
metrics={}
def call(name,*args):
    code=getattr(lib,'pipeline_'+name)(*args)
    if code:raise RuntimeError(lib.pipeline_error().decode())
def compare(name,a,b,tol=0):
    d=(a-b).abs();m=float(d.max());metrics[name]={'max_abs':m,'mean_abs':float(d.mean()),'different':int((a!=b).sum())}
    print(name,metrics[name],flush=True)
    assert torch.isfinite(a).all() and m<=tol,(name,m,tol)
def get(state,which):
    t=Tensor();call('view',state,which,ptr(t));shape=(t.channels,t.height,t.width) if t.layout==2 else (t.height,t.width,t.channels)
    a=torch.empty(shape,device='cuda');dst=view(a,t.layout==2);call('copy',ptr(t),ptr(dst),stream.cuda_stream);return a
with torch.cuda.stream(stream),torch.inference_mode():
    torch.manual_seed(742)
    h,w=320,384;nh,nw=320,448
    # Rows deliberately padded, all images nontrivial, fractional vector shifts.
    work_store=torch.rand((h,w+7,3),device='cuda');work=work_store[:,:w]
    motion_store=torch.empty((h,w+5,2),device='cuda');motion=motion_store[:,:w]
    yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
    motion[...,0]=2.375+torch.sin(yy*.043)*3;motion[...,1]=-.625+torch.cos(xx*.027)*2
    conf=torch.rand((h,w,1),device='cuda');conf[::7]=0
    state=lib.pipeline_create(w,h);assert state,lib.pipeline_error()
    old=None
    for frame in range(3):
        s=Settings(C.sizeof(Settings),[1,211,7][frame],[2.,.7,-.3][frame],[1.,.35,1.2][frame],[-1.,.25,-.8][frame],frame>0,.73,[1.,.64,.25][frame])
        call('prepare',state,ptr(view(work)),ptr(view(motion)),ptr(view(conf)),ptr(s),frame==0,stream.cuda_stream)
        current=nr.pad_color_for_neural_buffer(half_rtz(work)[None],nh,nw)
        mv=torch.zeros((1,nh,nw,2),device='cuda');gate=torch.zeros((1,nh,nw,1),device='cuda')
        if frame:
            pixels=(motion*motion.new_tensor([1/w,1/h]))*motion.new_tensor([w,h]);mv[0,:h,:w]=pixels
            px=xx+.5+pixels[...,0];py=yy+.5+pixels[...,1]
            visible=(px>=0)&(px<=w)&(py>=0)&(py<=h)
            gate[0,:h,:w]=(visible[...,None]*conf)*s.temporal_strength
        rt=nr.Block70RuntimeInputs(color=current,prev_output=current if old is None else old,mvec=mv,mvec_scale_xy=current.new_ones(2),output_dimensions_wh=torch.tensor([nw,nh]),style=s.style,local_structure_strength=s.structure,local_tone_strength=s.tone,skin_structure_strength=s.skin,use_auto_mask=bool(s.automatic_mask))
        packet=nr.build_preblock_features(rt,frame=frame)
        packet[:,7:10]=torch.where((gate>0).permute(0,3,1,2),packet[:,7:10],packet[:,4:7])
        compare(f'current_{frame}',get(state,1),current[0])
        uv=nr._output_uv(1,nh,nw,current.device);cu=nr._rect_uv(uv,current,None)
        sampled=nr.bilinear_texture(current,cu)
        hu=uv+nr.bilinear_texture(mv,cu)*current.new_tensor([1/nw,1/nh])
        prev=nr.catmull_rom_texture(rt.prev_output,hu,output_size=(nh,nw))
        compare(f'sampled_{frame}',get(state,2),sampled[0])
        compare(f'warped_{frame}',get(state,3),prev[0])
        compare(f'gate_{frame}',get(state,4),gate[0])
        compare(f'gaussian_{frame}',get(state,0)[:3],packet[0,:3],.002)
        compare(f'packet_rgb_{frame}',get(state,0)[4:10],packet[0,4:10],.000062)
        compare(f'packet_settings_{frame}',get(state,0)[10:],packet[0,10:])
        head=torch.randn((nh,nw,4),device='cuda')*.7
        call('composite',state,ptr(view(head)),stream.cuda_stream)
        result=nr.reconstruct_block70_color(head[None,...,:3],head[None,...,3:4],rt,gate)
        if s.intensity!=1:result=current+s.intensity*(result-current)
        compare(f'head_composite_{frame}',get(state,5),result[0],5e-5)
        call('commit',state);old=half_rtz(result)
        compare(f'half_history_{frame}',get(state,6),old[0],.000489)
        work=(work+.01*torch.randn_like(work)).clamp(0,1)
    # Explicit unsupported FSR/EASU and two-phase state discard.
    assert lib.pipeline_reconstruct(ptr(view(work)),ptr(view(work)),1,stream.cuda_stream)!=0
    assert 'unsupported' in lib.pipeline_error().decode()
    old_native=get(state,6).clone()
    s.intensity=0
    call('prepare',state,ptr(view(work)),ptr(view(motion)),ptr(view(conf)),ptr(s),1,stream.cuda_stream)
    call('composite',state,ptr(view(head)),stream.cuda_stream)
    compare('uncommitted_history',get(state,6),old_native)
    compare('intensity_zero',get(state,5),get(state,1))
    for oh,ow in [(127,173),(439,517),(320,384)]:
        for aa in [False,True]:
            dst=torch.empty((oh,ow,3),device='cuda');call('resize',ptr(view(work)),ptr(view(dst)),aa,stream.cuda_stream)
            expected=F.interpolate(work.permute(2,0,1)[None],size=(oh,ow),mode='bilinear',align_corners=False,antialias=aa)[0].permute(1,2,0)
            compare(f'resize_{oh}_{ow}_aa{aa}',dst,expected,8e-5)
    for enc,name in enumerate(['sRGB','linear','linear709_nits','PQ2020']):
        rgba=torch.rand((h,w+3,4),device='cuda')[:,:w];rgba[::5,:,3]=0
        if enc==2:rgba[...,:3]=(rgba[...,:3]-.1)*1400
        ctx=torch.empty((h,w,3),device='cuda');encoded=torch.empty_like(ctx);c=Color(enc,enc,203,1000)
        call('color',ptr(view(rgba)),ptr(view(ctx)),ptr(c),0,stream.cuda_stream)
        bridge=ColorBridge(ColorSettings(hdr=enc>=2,output_encoding=name))
        bchw=lambda a:a.permute(2,0,1)[None]
        expected=decode_input(bchw(rgba[...,:3]),name)
        compare(f'decode_{name}',ctx,expected[0].permute(1,2,0),.12 if enc==3 else 1e-5)
        call('color',ptr(view(ctx)),ptr(view(encoded)),ptr(c),1,stream.cuda_stream)
        # Compare bridge separately against its exact native context input.
        reference=bridge.encode(bchw(ctx));compare(f'encode_{name}',encoded,reference[0].permute(1,2,0),2e-6)
        modified=(encoded+.04*torch.sin(xx.float()*.02)[...,None]).clamp(0,1)
        protect=torch.rand((h,w,1),device='cuda');protect[::3]=1
        output=torch.empty((h,w+9,4),device='cuda')[:,:w]
        call('finish',ptr(view(ctx)),ptr(view(encoded)),ptr(view(modified)),ptr(view(rgba)),ptr(view(protect)),ptr(view(output)),ptr(c),.63,stream.cuda_stream)
        linear=composite(bchw(ctx),bchw(encoded),bchw(modified),bridge,None,OutputSettings(mix=.63),bchw(protect),bchw(rgba[...,3:4]))
        expected=bridge.output(linear)[0].permute(1,2,0)
        compare(f'finish_{name}',output[...,:3],expected,.0012 if enc>=2 else 2e-5)
        compare(f'alpha_{name}',output[...,3],rgba[...,3])
    from whitebox_pipeline.guides import GuideProcessor,GuideConfig
    lib.pipeline_guide_create.argtypes=[C.c_uint32,C.c_uint32];lib.pipeline_guide_create.restype=C.c_void_p
    lib.pipeline_guide_prepare.argtypes=[C.c_void_p,P,P,P,C.c_int,C.c_int,C.c_void_p]
    lib.pipeline_guide_view.argtypes=[C.c_void_p,C.c_int,P]
    lib.pipeline_guide_commit.argtypes=[C.c_void_p]
    lib.pipeline_guide_destroy.argtypes=[C.c_void_p]
    for luma in [False,True]:
        gs=lib.pipeline_guide_create(w,h);processor=GuideProcessor(GuideConfig(luma=luma))
        for frame in range(3):
            color=torch.rand((h,w,3),device='cuda');depth=torch.rand((h,w,1),device='cuda')*.04+.01
            if frame:color=torch.roll(last_color,(1,-2),(0,1))*.99;depth=torch.roll(last_depth,(1,-2),(0,1))
            call('guide_prepare',gs,ptr(view(color)),ptr(view(motion)),ptr(view(depth)),frame==0,luma,stream.cuda_stream)
            expected=processor.process(bchw(color).contiguous(),bchw(motion).contiguous(),bchw(depth).contiguous(),reset=frame==0)
            for which,key in enumerate(['motion','distrust','tests']):
                t=Tensor();call('guide_view',gs,which,ptr(t));dst=torch.empty((h,w,t.channels),device='cuda');call('copy',ptr(t),ptr(view(dst)),stream.cuda_stream)
                compare(f'guide_{luma}_{frame}_{key}',dst,expected[key][0].permute(1,2,0),0 if key!='tests' else 1e-5)
            call('guide_commit',gs);last_color=color;last_depth=depth
        stream.synchronize();lib.pipeline_guide_destroy(gs)
    # Residual transport on different output extents and cropped stride view.
    ref=torch.rand((439,517,3),device='cuda');base=half_rtz(work);result=work+.02*torch.randn_like(work);out=torch.empty_like(ref)
    call('transport',ptr(view(ref)),ptr(view(result)),ptr(view(base)),ptr(view(out)),stream.cuda_stream)
    expected=(bchw(ref)+F.interpolate(bchw(result-base),size=(439,517),mode='bilinear',align_corners=False)).clamp(0,1)
    compare('residual_transport',out,expected[0].permute(1,2,0),1e-6)
    stream.synchronize();lib.pipeline_destroy(state)
(ROOT/'native/pipeline/build/validation.json').write_text(json.dumps(metrics,indent=2))
print('PASS',len(metrics),'comparisons; non-default CUDA stream; strided 320x384 RGB/RGBA; FSR rejected')
