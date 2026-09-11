"""Vectorized decoding of the frozen model; no pickle or decoded-weight cache.
Original module constructors and BIN loading remain in use. Substitutions are
scoped to model construction, restored even on failure, and revision-gated.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path
import threading
import time
import numpy as np

FROZEN_SHA256='e796339b239f432fcedbefe823680befcafaf739e20ee2e29cfbf5c568984a1b'
_LOCK=threading.RLock()
_half=np.arange(2,dtype=np.int64)[:,None,None]
_row=np.arange(32,dtype=np.int64)[None,:,None]
_col=np.arange(8,dtype=np.int64)[None,None,:]
_SUBTILE_INDEX=(4*_col+(_row%16)//4)*16+_half*8+(_row&3)+(_row//16)*4

def _deposit_many(values,bits):
    result=np.zeros_like(values,dtype=np.int64)
    for bit,destination in enumerate(bits):result|=((values>>bit)&1)<<destination
    return result

@lru_cache(maxsize=4)
def _group_offsets(k_bits,n_bits,rep_bits):
    k=_deposit_many(np.arange(64,dtype=np.int64),k_bits)
    n=_deposit_many(np.arange(512,dtype=np.int64),n_bits).reshape(8,64)
    rep=_deposit_many(np.arange(4,dtype=np.int64),rep_bits)
    return rep[:,None,None,None]+k[None,None,:,None]+n[None,:,None,:]

def _same_bits(actual,expected):
    if isinstance(expected,np.ndarray):
        return isinstance(actual,np.ndarray) and actual.shape==expected.shape and actual.dtype==expected.dtype and np.array_equal(np.ascontiguousarray(actual).view(np.uint8),np.ascontiguousarray(expected).view(np.uint8))
    if isinstance(expected,(tuple,list)):
        return type(actual) is type(expected) and len(actual)==len(expected) and all(_same_bits(a,b) for a,b in zip(actual,expected))
    return actual==expected

@contextmanager
def decoding(module,*,audit=False):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=FROZEN_SHA256:
        raise ValueError('Vectorized loader only qualifies the frozen e796 model revision')
    with _LOCK:
        originals={name:getattr(module,name) for name in ['_unpack_qmma_b_e4m3','_decode_packed_qmma_projection','_decode_e4m3_numpy','_decode_group_factor']}
        route_descriptor=module.TorchPhysical8HFFN.__dict__['_route_bits']
        original_route=module.TorchPhysical8HFFN._route_bits
        table=module.decode_e4m3(np.arange(256,dtype=np.uint8))
        route_tables={bits:tuple(original_route(i,bits) for i in range(1<<bits)) for bits in (7,8)}
        report={'strategy':'vectorized frozen byte/layout decoders; original constructors; no persisted decoded-weight cache','model_source_sha256':FROZEN_SHA256,'audit':audit,'calls':{},'exact_audit_calls':{}}
        def unpack(raw):
            if len(raw)!=512:raise ValueError(f'QMMA-B subtile must be 512 bytes, got {len(raw)}')
            return np.frombuffer(raw,dtype=np.uint8)[_SUBTILE_INDEX]
        def decode(values):
            byte=np.asarray(values,dtype=np.uint8)
            if np.any((byte&127)==127):raise ValueError('unexpected NaN E4M3 value in block4 transition')
            return table[byte]
        def matrix(raw,start,input_width,output_width):
            count=input_width*output_width
            if start<0 or start+count>len(raw):raise ValueError(f'invalid packed projection {start}:{start+count}/{len(raw)}')
            if input_width<=0 or output_width<=0 or input_width%32 or output_width%16:
                return originals['_decode_packed_qmma_projection'](raw,start,input_width,output_width)
            tiles=np.frombuffer(raw,dtype=np.uint8,count=count,offset=start).reshape(input_width//32,output_width//16,512)
            fragments=tiles[...,_SUBTILE_INDEX]
            packed=fragments.transpose(0,3,1,2,4).reshape(input_width,output_width)
            return decode(packed)
        def factors(raw,base,k_positions,n_positions,rep_positions):
            offsets=_group_offsets(tuple(k_positions),tuple(n_positions),tuple(rep_positions))
            return table[np.frombuffer(raw,dtype=np.uint8)[base+offsets]]
        def route(value,bits):
            if bits in route_tables and 0<=value<len(route_tables[bits]):return route_tables[bits][value]
            return original_route(value,bits)
        replacements={'_unpack_qmma_b_e4m3':unpack,'_decode_packed_qmma_projection':matrix,'_decode_e4m3_numpy':decode,'_decode_group_factor':factors}
        def wrap(name,fn):
            def call(*args,**kwargs):
                result=fn(*args,**kwargs);report['calls'][name]=report['calls'].get(name,0)+1
                if audit:
                    # During the reference call restore every decoder, so the
                    # reference cannot accidentally call another optimized one.
                    active={key:getattr(module,key) for key in originals}
                    try:
                        for key,original in originals.items():setattr(module,key,original)
                        expected=originals[name](*args,**kwargs)
                    finally:
                        for key,current in active.items():setattr(module,key,current)
                    if not _same_bits(result,expected):raise AssertionError('Decoded model differs: '+name)
                    report['exact_audit_calls'][name]=report['exact_audit_calls'].get(name,0)+1
                return result
            return call
        try:
            for name,fn in replacements.items():setattr(module,name,wrap(name,fn))
            module.TorchPhysical8HFFN._route_bits=staticmethod(route)
            yield report
        finally:
            for name,original in originals.items():setattr(module,name,original)
            module.TorchPhysical8HFFN._route_bits=route_descriptor

def load_model(module,path,*,device='cuda',audit=False):
    start=time.perf_counter()
    with decoding(module,audit=audit) as report:model=module.load_model(path,device=device)
    report['load_seconds']=time.perf_counter()-start
    return model,report
