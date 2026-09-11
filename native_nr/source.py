"""Generate shape-specialized ownership tables from pure address codecs."""
from pathlib import Path
import shutil
import numpy as np

from . import layouts

KERNELS = Path(__file__).resolve().parent / 'cuda'


def array(name, values, kind='int'):
    values = np.asarray(values)

    def fmt(x):
        return '{' + ','.join(fmt(y) for y in x) + '}' if isinstance(x, list) else str(x)

    # Large shape ownership tables are module globals, not limited constant RAM.
    return '__device__ const ' + kind + ' ' + name + ''.join(
        f'[{n}]' for n in values.shape) + '=' + fmt(values.tolist()) + ';\n'


def classes(values):
    flat = values.reshape(len(values), -1)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    return unique.reshape(-1, *values.shape[1:]), inverse


def routes(heads, ds, up):
    pi, po = ds
    rows = np.where(pi[..., 0] >= 0, pi[..., 0] // (heads * 32), -1)
    unique, ids = classes(rows)
    ui, inverse, im = up
    projected = np.where(im[:, :, 0] >= 0, inverse[np.maximum(im[:, :, 0], 0), 0] // (heads * 32),
                         -1)
    local = np.full_like(projected, -1)
    global_rows = np.full((len(projected), 16), -1, np.int32)
    for tile, row in enumerate(projected):
        owned = np.unique(row[row >= 0])
        if len(owned) > 16:
            raise ValueError(
                f'Native UP{heads} CTA has {len(owned)} projection rows, expected <=16')
        global_rows[tile, :len(owned)] = ui[owned, 0] // 16
        for i, n in enumerate(owned):
            local[tile, row == n] = i
    up_unique, up_ids = classes(local)
    text = array('ds_ids', ids) + array('ds_rows', unique, 'signed char')
    text += array('up_ids', up_ids) + array('up_local', up_unique, 'signed char') + array(
        'up_global', global_rows)
    text += 'template<int First> __device__ __forceinline__ void collect_pool(Frag (&out)[4][4],u32 (&a)[4]) {\n int l=threadIdx.x,cid=ds_ids[blockIdx.x];\n'
    for first in (0, 2):
        text += f' if constexpr(First=={first}) {{\n'
        for word in range(4):
            text += f' {{ int row=l/4+{(word & 1)*8};\n'
            for pair in range(2):
                text += f' {{int k=route({word//2*16}+(l&3)*4+{pair*2});\n'
                for leaf in range(4):
                    banks = set()
                    for cls in unique:
                        for lane in range(32):
                            rr = int(cls[lane // 4 + (word & 1) * 8, leaf])
                            k = word // 2 * 16 + (lane & 3) * 4 + pair * 2
                            k = (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1)
                            if rr >= 0 and rr // 32 == first // 2:
                                banks.add((rr % 32 // 16) * 8 + (k // 8) * 2 + rr % 16 // 8)
                    text += f' int r{leaf}=ds_rows[cid][row][{leaf}]; u32 v{leaf}=0;\n'
                    text += f' {{ int rr=r{leaf},bank=(rr&16)/16*8+(k/8)*2+(rr&8)/8,src=(rr&7)*4+(k&7)/2;\n'
                    for bank in sorted(banks):
                        m, n, xy = first + bank // 8, bank % 8 // 2, 'y' if bank % 2 else 'x'
                        text += f' u32 b{bank}=__shfl_sync(0xffffffff,out[{m}][{n}].{xy},src); if(rr>=0 && rr/32=={first//2} && bank=={bank})v{leaf}=b{bank};\n'
                    text += ' }\n'
                text += f' if((r0>=0 && r0/32=={first//2}) || (r1>=0 && r1/32=={first//2}) || (r2>=0 && r2/32=={first//2}) || (r3>=0 && r3/32=={first//2})) {{ half2 mean=__hmul2(__hadd2(__hadd2(h2(v0),h2(v1)),__hadd2(h2(v2),h2(v3))),constant2(.25f)); a[{word}] |= u32(__nv_cvt_halfraw2_to_fp8x2((__half2_raw)mean,__NV_SATFINITE,__NV_E4M3))<<{pair*16}; }} }}\n'
            text += ' }\n'
        text += ' }\n'
    return text + '}\n'


def prepare(shape, directory):
    directory = Path(directory)
    # A shape owns its own generated sources; preparing another shape cannot
    # overwrite this one's include files or invalidate its module cache identity.
    directory.mkdir(parents=True, exist_ok=True)
    for filename in sorted(set(OUTER_UNITS.values())):
        shutil.copyfile(KERNELS / filename, directory / filename)
    ds, up = {}, {}
    for heads, block in ((2, 8), (4, 14), (8, 22)):
        p = shape.outer(block)
        ds[heads] = layouts.pool_metadata(p, layouts.FAMILY, layouts.TWO)
    for heads, block in ((2, 62), (4, 56), (8, 48)):
        p = shape.outer(block)
        ui, inverse = layouts.up_metadata(p, layouts.FAMILY, layouts.TWO)
        im, _ = (layouts.maps2 if heads == 2 else layouts.wide_maps)(p)
        up[heads] = ui, inverse, im
    for heads in (4, 8):
        (directory / f'routes{heads}.cuh').write_text(routes(heads, ds[heads], up[heads]))
    text = 'namespace transition_packet {\n'
    for heads in (2, 4, 8):
        po = ds[heads][1]
        text += array(f'pixels{heads}', np.where(po[:, :, 0] >= 0, po[:, :, 0] // 16, -1))
    counts = {
        2: shape.height // 8 * (shape.width // 8),
        4: shape.height // 16 * (shape.width // 16),
        8: shape.deep_hw[0] * shape.deep_hw[1]
    }
    text += '__device__ __forceinline__ int pixel(int H,int tile,int row){return H==2?pixels2[tile][row]:H==4?pixels4[tile][row]:pixels8[tile][row];}\n'
    text += f'__device__ __forceinline__ int count(int H){{return H==2?{counts[2]}:H==4?{counts[4]}:{counts[8]};}}\n'
    text += '__device__ __forceinline__ int planar(int p,int c,int n){return p*16+(c/16)*n*16+(c&1)+((c&6)<<1)+((c&8)>>2);}\n}\n'
    (directory / 'addresses.cuh').write_text(text)
    # Templates already contain the original symbolic shape substitutions.
    return ds, up


# Logical roles share functional translation units; launch order is unchanged.
OUTER_UNITS = {
    'shallow_static_geometry/one.cu': 'shallow.cu',
    'shallow_static_geometry/transition.cu': 'shallow.cu',
    'post_static_packet/post.cu': 'shallow.cu',
    'two_physical_seed/two.cu': 'heads2.cu',
    'wide_transition_packet/two.cu': 'heads2.cu',
    'wide_input_packet/wide4.cu': 'heads4.cu',
    'wide_transition_packet/ds4.cu': 'heads4.cu',
    'wide_transition_packet/up4.cu': 'heads4.cu',
    'eight_full_projection/eight.cu': 'heads8.cu',
    'wide_transition_packet/ds8.cu': 'heads8.cu',
    'wide_transition_packet/up8.cu': 'heads8.cu',
    'utility.cu': 'utility.cu'
}
DEEP_UNITS = {
    'sixteen': 'deep16.cu',
    'local': 'deep16.cu',
    'matrix': 'vit.cu',
    'qkv': 'vit.cu',
    'attention': 'vit.cu',
    'transition': 'bridge.cu',
    'repack': 'bridge.cu',
    'split': 'bridge.cu'
}
