"""Per-kernel private NVRTC12.8 compiler. Ordinary kernels retain their compiler."""
from runtime import bootstrap
import contextvars,contextlib,ctypes as C,importlib,os,threading
from pathlib import Path
_enabled=contextvars.ContextVar('dlss5_fp8_compiler',default=False)
_lock=threading.RLock();_installed=False;_compiler=None

class PrivateCompiler:
 def __init__(self):
  self.root=Path(os.environ.get('DLSS5_FP8_TOOLCHAIN',str(bootstrap.ROOT/'.toolchains/cuda12.8')))
  dll=self.root/'nvrtc/bin/nvrtc64_120_0.dll'
  if not dll.is_file():raise RuntimeError('Private FP8 compiler missing. Run setup_fp8_toolchain.py or use --no-fp8-one. No automatic download.')
  self.directory=os.add_dll_directory(str(dll.parent))
  self.builtins=C.WinDLL(str(dll.parent/'nvrtc-builtins64_128.dll'))
  self.dll=C.WinDLL(str(dll))
  def bind(name,types):
   fn=getattr(self.dll,name);fn.argtypes=types;fn.restype=C.c_int;return fn
  self.version=bind('nvrtcVersion',[C.POINTER(C.c_int),C.POINTER(C.c_int)])
  major=C.c_int();minor=C.c_int();self.check(self.version(C.byref(major),C.byref(minor)))
  if (major.value,minor.value)<(12,8):raise RuntimeError('FP8 F16 accumulation requires NVRTC12.8+')
  self.create=bind('nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
  self.compile=bind('nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
  self.logsize=bind('nvrtcGetProgramLogSize',[C.c_void_p,C.POINTER(C.c_size_t)])
  self.log=bind('nvrtcGetProgramLog',[C.c_void_p,C.c_void_p])
  self.size=bind('nvrtcGetCUBINSize',[C.c_void_p,C.POINTER(C.c_size_t)])
  self.cubin=bind('nvrtcGetCUBIN',[C.c_void_p,C.c_void_p])
  self.destroy=bind('nvrtcDestroyProgram',[C.POINTER(C.c_void_p)])
 @staticmethod
 def check(code):
  if code:raise RuntimeError(f'Private NVRTC error {code}')
 def __call__(self,code,target_format='cubin',arch=None,options=None,verbose=False):
  if target_format!='cubin' or arch not in (None,89):raise ValueError('Private compiler is qualified only for SM89 CUBIN')
  opts=['-default-device','--gpu-architecture=sm_89','-I'+str(self.root/'runtime/include'),'-I'+str(self.root/'cccl/include'),'-I'+str(self.root/'cccl/include/cuda/std')]
  opts+=([options] if isinstance(options,str) else list(options or []))
  source=('#include <tl_templates/cuda/nvrtc_std.h>\n'+code).encode();program=C.c_void_p()
  self.check(self.create(C.byref(program),source,b'fp8_block1',0,None,None))
  try:
   args=(C.c_char_p*len(opts))(*(x.encode() for x in opts));status=self.compile(program,len(opts),args)
   if status:
    n=C.c_size_t();self.check(self.logsize(program,C.byref(n)));log=C.create_string_buffer(n.value);self.check(self.log(program,log));raise RuntimeError(log.value.decode('utf-8','replace'))
   n=C.c_size_t();self.check(self.size(program,C.byref(n)));data=C.create_string_buffer(n.value);self.check(self.cubin(program,data));return bytearray(data.raw)
  finally:self.destroy(C.byref(program))

def install():
 global _installed
 with _lock:
  if _installed:return
  lib=importlib.import_module('tilelang.jit.adapter.nvrtc.libgen');ordinary=lib.compile_cuda
  # Bind the ordinary loader before loading another same-named NVRTC DLL.
  from tilelang.contrib.nvrtc import get_nvrtc_version
  get_nvrtc_version()
  def dispatch(*a,**k):
   global _compiler
   if not _enabled.get():return ordinary(*a,**k)
   with _lock:
    if _compiler is None:_compiler=PrivateCompiler()
    return _compiler(*a,**k)
  lib.compile_cuda=dispatch;_installed=True

@contextlib.contextmanager
def private_compile():
 install();token=_enabled.set(True)
 try:yield
 finally:_enabled.reset(token)
