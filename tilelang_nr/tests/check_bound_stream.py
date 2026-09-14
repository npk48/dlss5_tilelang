"""GPU regression for prepared TileLang argument storage and caller streams."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TILELANG_CACHE_DIR','C:/tmp/d5-tl-eval/dynamic/cache')
import torch
from tilelang_nr.kernels.utility import _clear
from tilelang_nr.common.runtime_jit import runtime_compilation_stats

def main():
    kernel=_clear(777)
    initial=runtime_compilation_stats()['jit_builds']
    for stream in (torch.cuda.Stream(),torch.cuda.Stream(),torch.cuda.default_stream()):
        with torch.cuda.stream(stream):
            counters=torch.full((777,),7,device='cuda',dtype=torch.int32)
            status=torch.ones(1,device='cuda',dtype=torch.int32)
            kernel(counters,status)
        stream.synchronize()
        assert bool((counters == -1).all()) and status.item()==0
        assert all(int(record[0].hStream)==stream.cuda_stream for record in kernel._prepared.records)
    assert runtime_compilation_stats()['jit_builds']==initial
    print('PASS: refreshed packet pointers and two non-default streams plus default stream; no extra JIT builds')
if __name__=='__main__':main()
