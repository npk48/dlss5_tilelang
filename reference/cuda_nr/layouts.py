"""Packet geometry and physical outer-buffer extents, not an image resizing policy."""
from dataclasses import dataclass


def ceil_to(value, unit):
    return (value + unit - 1) // unit * unit


def neural_size(height, width):
    return ceil_to(max(height, 320), 64), ceil_to(max(width + 64, 320), 64)


@dataclass(frozen=True)
class Shape:
    height: int
    width: int

    def __post_init__(self):
        if self.height < 64 or self.width < 64 or self.height % 64 or self.width % 64:
            raise ValueError(
                f'Native NR expects an already prepared 64-aligned packet, got {self.height}x{self.width}'
            )

    @property
    def deep_hw(self):
        # DS14 crosses axes in the logical view; DS22 pads each native axis to 8
        # before /2. These are real model descriptor extents, not input padding.
        return ceil_to(self.width // 16, 8) // 2, ceil_to(self.height // 16, 8) // 2

    def outer(self, block):
        if block <= 4 or 66 <= block <= 69:
            heads, divisor = 1, 2
        elif block <= 8 or 62 <= block <= 65:
            heads, divisor = 2, 4
        elif block <= 14 or 56 <= block <= 61:
            heads, divisor = 4, 8
        else:
            heads, divisor = 8, 16
        h, w = self.height // divisor, self.width // divisor
        base = 1 if block <= 4 else 5 if block <= 8 else 9 if block <= 14 else 15 if block <= 22 else 48 if block <= 55 else 54 if block <= 61 else 62 if block <= 65 else 66
        sx, sy = ((0, 0), (-4, -4), (-4, 0), (0, -4))[(block - base) % 4]
        gx, gy = (w - sx + 7) // 8, (h - sy + 7) // 8
        seq = block + (2 if block <= 22 else 85)
        return dict(block=block,
                    seq=seq,
                    heads=heads,
                    shape=(h, w),
                    shift=(sx, sy),
                    grid=(gx, gy, 1),
                    pool_shape=(ceil_to(h, 8) // 2, ceil_to(w, 8) // 2) if heads == 8 else
                    (h // 2, w // 2))

    def describe(self):
        dh, dw = self.deep_hw
        return {
            'packet_hw': [self.height, self.width],
            'deep_input_hw': [dh, dw],
            'vit_hw': [ceil_to(dh // 2, 4), ceil_to(dw // 2, 4)],
            'head_hw': [self.height, self.width]
        }


# Outer 1H/2H/4H/8H address maps.
"""Pure address algebra for native outer stages; no filesystem/model/capture imports."""
import numpy as np

P = np.array([
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 20, 21, 24, 25, 28, 29, 18, 19,
    22, 23, 26, 27, 30, 31
], np.int32)


def tin(y, x, c, w):
    t = (y & 7) * 8 + (x & 7)
    p = 64 * (t // 2) + 4 * (t & 1) + 8 * (c // 4) + (c & 3)
    s = p // 512
    cy = (y // 8) * 2 + ((s == 1) | (s == 2))
    cx = (x // 8) * 2 + ((s == 2) | (s == 3))
    return (cy * (w // 4) + cx) * 512 + (p & 511)


def planar(pixel, c, count):
    return pixel * 16 + (c // 16) * count * 16 + (c & 1) + ((c & 6) << 1) + ((c & 8) >> 2)


def raw2(y, x, c, h, w):
    t = (y % 8) * 8 + x % 8
    return ((c // 16) * (h // 8) * (w // 4) + (y // 8) * (w // 4) +
            (x // 8) * 2 + t // 32) * 512 + (t % 32) * 16 + c % 16


def generic2(p):
    h, w = p['shape']
    gx, gy, _ = p['grid']
    sx, sy = p['shift']
    t, c = np.indices((64, 64), dtype=np.int64)
    lab = (c & 3) | ((c & 12) << 2) | ((c & 16) >> 1) | ((c & 32) << 4) | ((t & 7) << 6) | (
        (t & 8) >> 1) | ((t & 48) << 6)
    assert np.array_equal(np.sort(lab.ravel()), np.arange(4096))
    by, bx = np.indices((gy, gx), dtype=np.int64)
    by = by.reshape(-1, 1, 1)
    bx = bx.reshape(-1, 1, 1)
    qy = (8 * by + sy) // 4 + lab // 2048
    qx = (8 * bx + sx) // 4 + (lab % 2048) // 1024
    valid = (qy >= 0) & (qy < h // 4) & (qx >= 0) & (qx < w // 4)
    im = (qy * (w // 4) + qx) * 1024 + ((lab % 2048) // 512 % 2) * 512 + lab % 512
    im = np.where(valid, im, -1).astype('<i4')
    om = im[:, :, np.r_[P, P + 32]].copy()
    return im, om


def maps2(p):
    h, w = p['shape']
    im, om = generic2(p)
    if p['seq'] == 7:
        gx, gy, _ = p['grid']
        by, bx, m, k = np.indices((gy, gx, 64, 64), dtype=np.int64)
        # first2H independent key/query route, followed by inpview M->pixel.
        t = (m & 1) | ((m & 2) << 3) | ((m & 4) >> 1) | ((m & 8) >> 1) | ((m & 16) << 1) | (
            (m & 32) >> 2)
        pixel = (t & 1) | ((t & 14) << 2) | ((t >> 3) & 6)
        row = (by * 8 + pixel // 8) * w + bx * 8 + pixel % 8
        im = (row * 16 + (k // 16) * h * w * 16 + k % 16).reshape(-1, 64, 64).astype('<i4')
        # publish_first_2h exact full-address permutation, not a local roll.
        d = np.arange(h * w * 64, dtype=np.int64)
        outer = d >> 10
        source = (outer // (w // 4)) * (w * 64) + (outer % (w // 4)) * 64 + (d & 3) + (
            (d >> 4) & 15) * 4 + ((d >> 2) & 1) * w * 32 + ((d >> 3) & 1) * h * w * 16 + (
                (d >> 8) & 1) * w * 16 + ((d >> 9) & 1) * h * w * 32
        assert np.array_equal(np.sort(source), d)
        inverse = np.empty_like(source)
        inverse[source] = d
        om = inverse[im[:, :, np.r_[P, P + 32]]].astype('<i4')
    if p['seq'] == 150:
        # outview is consumed as physical decoder token/output N planes.
        token, n = decoder_internal_raw(h, w)
        inverse = np.empty(h * w * 64, np.int32)
        yy, xx, cc = np.indices((h, w, 64), dtype=np.int64)
        inverse[raw2(yy, xx, cc, h, w).ravel()] = planar(decoder_pixel(token, w), n, h * w).ravel()
        mask = om >= 0
        om = om.copy()
        om[mask] = inverse[om[mask]]
    for a in (im, om):
        assert np.array_equal(np.sort(a[a >= 0]),
                              np.arange(h * w * 64)), ('map2 bijection', p['seq'])
    return im, om


def decoder_pixel(token, w):
    # physical_2h_block._window_layout: fused token -> native spatial coords.
    low = token & 63
    outer = token >> 6
    row = (outer // (w // 16)) * 4 + ((low >> 3) & 1) + 2 * (low & 1)
    col = (outer % (w // 16)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) + 4 * (
        (low >> 4) & 1) + 8 * ((low >> 5) & 1)
    return row * w + col


def decoder_internal_raw(h, w):
    r, c, n = np.indices((h, w, 64), dtype=np.int64)
    on = (n & 1) | ((c & 1) << 1) | (((c >> 1) & 1) << 2) | (((n >> 1) & 1) << 3) | ((
        (n >> 3) & 1) << 4) | (((r >> 2) & 1) << 5)
    low = ((n >> 2) & 1) | (((c >> 2) & 1) << 1) | ((r & 1) << 2) | (((r >> 1) & 1) << 3) | ((
        (c >> 3) & 1) << 4)
    t = low + 32 * (c // 16 + (w // 16) * (r // 8 + (h // 8) * (n // 16)))
    return t, on


def pool1_maps(p, r, wts):
    h, w = p['shape']
    oh, ow = h // 2, w // 2
    y, x, k = np.indices((oh, ow, 32), dtype=np.int64)
    parts = []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        yy = 2 * y + dy
        xx = 2 * x + dx
        slot = r['ptoc'][(yy % 8) * 8 + xx % 8]
        yy = yy // 8 * 8 + slot // 8
        xx = xx // 8 * 8 + slot % 8
        parts.append(tin(yy, xx, wts['cp'][k], w))
    im = np.stack(parts, axis=2).reshape(-1, 4, 32).astype('<i4')
    om = planar(np.arange(oh * ow)[:, None], np.arange(64)[None, :], oh * ow).astype('<i4')
    return im, om


def pool2_maps(p):
    h, w = p['shape']
    oh, ow = h // 2, w // 2
    t, c = np.indices((64, 64), dtype=np.int64)
    n = (c & 1) | ((t & 1) << 1) | ((t & 2) << 1) | ((c & 2) << 2) | ((c & 8) << 1) | (t & 32)
    m = ((t & 4) >> 2) | ((t & 16) >> 3) | (c & 4) | ((c & 16) >> 1) | ((t & 8) << 1) | (c & 32)
    k = np.r_[P, P + 32][n]
    inverse = np.empty(4096, np.int32)
    inverse[(m * 64 + k).ravel()] = (t * 64 + c).ravel()
    y, x, j, k = np.indices((oh, ow, 4, 64), dtype=np.int64)
    mi = ((y % 4) * 4 + x % 4) * 4 + j
    tc = inverse[mi * 64 + k]
    tt = tc // 64
    cc = tc % 64
    yy = y // 4 * 8 + tt // 8
    xx = x // 4 * 8 + tt % 8
    im = raw2(yy, xx, cc, h, w).reshape(-1, 4, 64).astype('<i4')
    # validate_native_routes.raw4(first=True): canonical compact pixels are
    # odd-half-band ordered, not row-major in the following inpview's planes.
    yy, xx = np.indices((oh, ow), dtype=np.int64)
    ty, tx, ly, lx = yy // 4, xx // 4, yy % 4, xx % 4
    band = ty // 2 + (oh // 8) * ((lx >> 1) + 2 * (ly >> 1))
    pixel = tx * 2 + (ty & 1) * (ow // 2) + (lx & 1) * ow + (ly & 1) + band * (2 * ow)
    om = planar(pixel.reshape(-1, 1), np.arange(128)[None, :], oh * ow).astype('<i4')
    return im, om


def deposit_outer(x, bits):
    return sum(((x >> i) & 1) << b for i, b in enumerate(bits))


def canonical4_out(h, w):
    r, c, n = np.indices((h, w, 128), dtype=np.int64)
    ty, tx, ly, lx = r // 4, c // 4, r % 4, c % 4
    y = ty // 2 + (h // 8) * ((lx >> 1) + 2 * (ly >> 1))
    st = (y & 1) + ((ly & 1) << 1) + ((tx & 1) << 2) + ((lx & 1) << 3) + 16 * ((tx // 2) +
                                                                               (w // 8) *
                                                                               ((ty & 1) + 2 *
                                                                                (y // 2)))
    # physical output raw4 N bits (0,4,5,1,3,9,10), then outview raw->planar.
    bits = (0, 4, 5, 1, 3, 9, 10)
    raw = deposit_outer(n, bits) + deposit_outer(st, tuple(b for b in range(24) if b not in bits))
    yy, xx, nn = np.indices((h, w, 128), dtype=np.int64)
    nc = (nn & 1) | ((nn & 6) << 3) | ((nn & 8) >> 2) | ((nn & 16) >> 1) | ((nn & 224) << 4)
    ss = ((yy >> 1) & 1) | ((xx & 3) << 1) | ((yy & 1) << 3)
    addr = nc + ((ss & 1) << 2) + ((ss & 14) << 5) + (xx // 4 + (w // 4) * (yy // 4)) * 2048
    inverse = np.empty(h * w * 128, np.int32)
    inverse[addr.ravel()] = planar(yy * w + xx, nn, h * w).ravel()
    return inverse[raw]


def up2_maps(h, w):
    ih, iw = h // 2, w // 2
    r, c = np.indices((ih, iw), dtype=np.int64)
    ty, tx, ly, lx = r // 4, c // 4, r % 4, c % 4
    y = ty // 2 + (ih // 8) * ((lx >> 1) + 2 * (ly >> 1))
    st = (y & 1) | ((ly & 1) << 1) | ((tx & 1) << 2) | ((lx & 1) << 3)
    token = st + 16 * ((tx // 2) + (iw // 8) * ((ty & 1) + 2 * (y >> 1)))
    raw = canonical4_out(ih, iw)
    im = np.empty((ih * iw, 128), np.int32)
    im[token.ravel()] = raw.reshape(-1,
                                    128)[:,
                                         np.argsort(np.concatenate([P + 32 * g for g in range(4)]))]
    r, c, n = np.indices((h, w, 64), dtype=np.int64)
    y = r // 16 + (h // 16) * (n >> 4)
    low = (y & 1) | ((r & 1) << 1) | (((c >> 3) & 1) << 2) | (((n >> 2) & 1) << 3)
    st = low + 16 * (c // 16 + (w // 16) * (((r >> 3) & 1) + 2 * (y >> 1)))
    sn = (n & 1) | (((c >> 1) & 1) << 1) | (((n >> 1) & 1) << 2) | ((c & 1) << 3) | ((
        (n >> 3) & 1) << 4) | (((r >> 2) & 1) << 5)
    pc = np.r_[P, P + 32][sn]
    gate = (n & 1) | ((c & 1) << 1) | (((c >> 1) & 1) << 2) | (((n >> 1) & 1) << 3) | ((
        (n >> 3) & 1) << 4) | (((r >> 2) & 1) << 5)
    dest = raw2(r, c, n, h, w)
    return im, np.stack((st * 64 + pc, dest, gate), axis=-1).reshape(-1, 3).astype('<i4')


def up1_maps(h, w, routes):
    ih, iw = h // 2, w // 2
    im = planar(decoder_pixel(np.arange(ih * iw)[:, None], iw),
                np.argsort(np.r_[P, P + 32])[None, :], ih * iw).astype('<i4')
    y, x, c = np.indices((h, w, 32), dtype=np.int64)
    t = (((y >> 1) ^ (y >> 2)) & 1) | (((x >> 2) & 1) << 1) | (((y >> 2) & 1) << 2) | (
        (x & 1) << 3) | (((x >> 3) & 1) << 4) | (((x >> 4) & 1) << 5)
    t += 64 * (x // 32 + (w // 32) * (y // 8))
    n = (c & 1) | (((c >> 4) & 1) << 1) | (((c >> 1) & 1) << 2) | (c & 8) | (((c >> 2) & 1) << 4)
    return im, np.stack((t * 32 + P[n], tin(y, x, c, w), c), axis=-1).reshape(-1, 3).astype('<i4')


def wide_planar(pixel, channel, count):
    lo = (channel & 1) | ((channel & 6) << 1) | ((channel & 8) >> 2)
    return pixel * 16 + (channel // 16) * count * 16 + lo


def blocked(heads, y, x, n, width):
    # Native scalar STG addresses: 4H 4x4/C128, 8H 4x8/C256.
    nc = (n & 1) | ((n & 6) << 3) | ((n & 8) >> 2) | ((n & 16) >> 1) | ((n & 224) << 4)
    st = ((y >> 1) & 1) | ((x & 3) << 1) | ((y & 1) << 3)
    if heads == 4:
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + (x // 4 + (width // 4) * (y // 4)) * 2048
    st |= (x & 4) << 2
    return nc + ((st & 1) << 2) + ((st & 14) << 5) + ((st & 16) << 8) + (x // 8 +
                                                                         (((width + 7) // 8)) *
                                                                         (y // 4)) * 8192


def wide_maps(launch):
    h, w = launch['shape']
    heads = launch['heads']
    gx, gy, _ = launch['grid']
    by, bx, slot, n = np.indices((gy, gx, 64, heads * 32), dtype=np.int64)
    if heads == 4:
        y = by * 8 + ((slot >> 4) & 3) * 2 + (slot & 1)
        x = bx * 8 + ((slot >> 1) & 7)
    else:
        y = by * 8 + slot // 8
        x = bx * 8 + (slot & 7)
    x += launch['shift'][0]
    y += launch['shift'][1]
    active = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    raw = blocked(heads, y, x, n, w)
    plane = planar(y * w + x, n, h * w)
    im = plane if launch['seq'] in (11, 17) else raw
    om = plane if launch['seq'] in (140, 146) else raw
    im = np.where(active, im, -1).reshape(-1, 64, heads * 32).astype(np.int32)
    om = np.where(active, om, -1).reshape(im.shape).astype(np.int32)
    for m in (im, om):
        a = m[m >= 0]
        assert len(a) == h * w * heads * 32 and len(np.unique(a)) == len(a), ('map injection',
                                                                              launch['seq'])
    return im, om


def canonical4(h, w):
    # Generalized odd-half-band formula, physical_4h_block / raw4.
    r, c, n = np.indices((h, w, 128), dtype=np.int64)
    ty, tx, ly, lx = r // 4, c // 4, r % 4, c % 4
    y = ty // 2 + (h // 8) * ((lx >> 1) + 2 * (ly >> 1))
    st = (y & 1) + ((ly & 1) << 1) + ((tx & 1) << 2) + ((lx & 1) << 3) + 16 * ((tx // 2) +
                                                                               (w // 8) *
                                                                               ((ty & 1) + 2 *
                                                                                (y // 2)))
    bits = (0, 4, 5, 1, 3, 9, 10)
    return deposit_outer(n, bits) + deposit_outer(st, tuple(b for b in range(24) if b not in bits))


def pool_maps(launch):
    h, w = launch['shape']
    heads = launch['heads']
    oh, ow = launch['pool_shape']
    n = np.arange(heads * 64)[None, :]
    if heads == 4:
        raw = canonical4(h, w)
        # Vertical Half pairs, then pair sum * .25; no intermediate E4.
        im = np.stack((raw[0::2, 0::2], raw[1::2, 0::2], raw[0::2, 1::2], raw[1::2, 1::2]),
                      axis=2).reshape(-1, 4, 128)
        r, c = np.indices((oh, ow), dtype=np.int64)
        pixel = ((r % 2) * 2 + c % 2) * (oh * ow // 4) + (r // 2) * (ow // 2) + c // 2
        om = planar(pixel.reshape(-1, 1), n, oh * ow)
    else:
        y, x, k = np.indices((oh, ow, 256), dtype=np.int64)
        parts = []
        # physical_8h_block local axes: pair local_x, then local_y.
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            yy = y * 2 + dy
            xx = x * 2 + dx
            parts.append(np.where((yy < h) & (xx < w), blocked(8, yy, xx, k, w), -1))
        im = np.stack(parts, axis=2).reshape(-1, 4, 256)
        om = planar(np.arange(oh * ow)[:, None], n, oh * ow)
    return im.astype(np.int32), om.astype(np.int32)


def up_maps(launch):
    h, w = launch['shape']
    heads = launch['heads']
    k = np.arange(heads * 64)[None, :]
    if heads == 4:
        r, c = np.indices((h, w), dtype=np.int64)
        quarter = ((r // 4) * (w // 4) + c // 4) * 4 + ((r >> 1) & 1) * 2 + ((c >> 1) & 1)
        count = h * w // 4
        pixel = (quarter % 4) * (count // 4) + quarter // 4
        im = planar(pixel.reshape(-1, 1), k, count)
        om = canonical4(h, w).reshape(-1, 128)
    else:
        # SM89 mixed48 source is compact planar 512 at rounded half extent.
        y, x = np.indices((h, w), dtype=np.int64)
        oh, ow = ((h + 7) // 8) * 4, ((w + 7) // 8) * 4
        im = planar(((y // 2) * ow + x // 2).reshape(-1, 1), k, oh * ow)
        om = blocked(8, y[:, :, None], x[:, :, None],
                     np.arange(256)[None, None, :], w).reshape(-1, 256)
    return im.astype(np.int32), om.astype(np.int32)


def pool_metadata(p, f, o):
    H = p['heads']
    K = H * 32
    N = K * 2
    _, om = (o['maps2'] if H == 2 else f['maps'])(p)
    pi, po = (o['pool2_maps'] if H == 2 else f['pool_maps'])(p)
    inverse = np.full(max(int(om.max()) + 1, np.prod(p['shape']) * K), -1, np.int32)
    flat = om.reshape(-1)
    valid = flat >= 0
    inverse[flat[valid]] = np.nonzero(valid)[0]
    # This is the proven full-K CTA ownership inverse, not spatial averaging.
    sample = np.max(pi.reshape(len(pi), -1), axis=1)
    owner = np.where(sample >= 0, inverse[np.maximum(sample, 0)] // (64 * K), -1)
    tiles = len(om)
    ip = np.full((tiles, 16, 4, K), -1, np.int32)
    op = np.full((tiles, 16, N), -1, np.int32)
    counts = np.zeros(tiles, np.int32)
    for row, tile in enumerate(owner):
        if tile < 0:
            continue  # Original pool8_padding is retained for all-pad rows.
        slot = counts[tile]
        assert slot < 16, (p['seq'], int(tile), int(slot))
        counts[tile] += 1
        ids = pi[row]
        ip[tile, slot] = np.where(ids >= 0, inverse[np.maximum(ids, 0)] % (64 * K), -1)
        op[tile, slot] = po[row]
    return ip, op


def up_metadata(p, f, o):
    H = p['heads']
    N = H * 32
    h, w = p['shape']
    inverse = np.zeros((h * ((w + 7) // 8 * 8 if H == 8 else w) * N, 2), np.int32)
    if H == 2:
        ui, fm = o['up2_maps'](h, w)
        inverse[fm[:, 1], 0] = fm[:, 0]
        inverse[fm[:, 1], 1] = fm[:, 2]
    else:
        source, uo = f['up_maps'](p)
        ui, which = np.unique(source, axis=0, return_inverse=True)
        inverse[uo.reshape(-1), 0] = (which[:, None] * N + np.arange(N)).reshape(-1)
        inverse[uo.reshape(-1), 1] = np.tile(np.arange(N), len(uo))
    return np.ascontiguousarray(ui, dtype=np.int32), inverse


FAMILY = {'maps': wide_maps, 'pool_maps': pool_maps, 'up_maps': up_maps}
TWO = {'maps2': maps2, 'pool2_maps': pool2_maps, 'up2_maps': up2_maps}

# Deep 16H and ViT address maps.
"""Integer-only raw address algebra. No captured data or model execution."""
import numpy as np


def align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def dep(v, bits):
    out = np.zeros_like(v, dtype=np.int64)
    for i, b in enumerate(bits):
        out |= ((v >> i) & 1) << b
    return out


def raw16(h, w, first=False, decoder=False, pool=False, channels=512):
    y, x, n = np.indices((h, w, channels), dtype=np.int64)
    cb = (0, 4, 5, 1, 3, 9, 10, 11, 12, 13)[:(channels - 1).bit_length()]
    if first:
        pixel = (x // 4 * 4 + y % 4) * h + (y // 4 * 4 + x % 4)
        return (pixel * 16 + (n // 16) * h * w * 16 + dep(n % 16, (0, 2, 3, 1))).reshape(-1)
    if decoder:
        token = y * w + x
    elif pool:
        token = ((x >> 1) & 1) + ((x & 1) << 1) + (((y >> 1) & 1) << 2) + (
            (y & 1) << 3) + 16 * (y // 4 + (h // 4) * (x // 4))
    else:
        token = ((y >> 1) & 1) + ((x & 3) << 1) + ((y & 1) << 3) + 16 * (y // 4 + (h // 4) *
                                                                         (x // 4))
    return (dep(n, cb) + dep(token, tuple(b for b in range(32) if b not in cb))).reshape(-1)


def encoder_pixels(h, w):
    t = np.arange(h * w, dtype=np.int64)
    within = t % (h * 4)
    slot = within % 32
    y = (within // 32) * 8 + ((slot >> 3) & 1) + ((slot & 1) << 1) + (((slot >> 4) & 1) << 2)
    x = (t // (h * 4)) * 4 + ((slot >> 1) & 3)
    return y * w + x


def pointmap(h, w, channels, first=False):
    y, x = np.indices((h, w), dtype=np.int64)
    m = (((x & ~1) | (y & 1)) * h + ((y & ~1) | (x & 1))).reshape(-1, 1)
    m = (m // 16) * 16 + 2 * (m & 7) + ((m >> 3) & 1)
    n = np.arange(channels, dtype=np.int64)[None, :]
    cb = (0, 1, 4, 5, 3, 9, 10, 11, 12, 13, 14, 15) if first else (0, 4, 5, 1, 3, 9, 10, 11, 12, 13,
                                                                   14, 15)
    return ((m // 16) * (16 * channels) + dep(m % 16, (2, 6, 7, 8)) +
            dep(n, cb[:(channels - 1).bit_length()])).reshape(-1)


def halfmap(point):
    return dep(point, (0, 2, 1, 8, 3, 4, 5, 6, 7, *range(9, 32)))


def repack99(h, w):
    m = np.arange(h * w, dtype=np.int64)[:, None]
    j = np.arange(256, dtype=np.int64)[None, :]
    x, y = m // h, m % h
    out_m = 16 * ((x // 4) * (h // 4) + (y // 4)) + 4 * (x % 4) + y % 4

    def word(t):
        return 4096 * (t // 16) + 16 * (t & 7) + (
            (t >> 3) & 1) + 128 * (j // 8) + 4 * (j & 3) + 2 * ((j >> 2) & 1)

    source = (word(m)[..., None] * 4 + np.arange(4)).reshape(-1)
    target = (word(out_m)[..., None] * 4 + np.arange(4)).reshape(-1)
    mapping = np.empty(source.size, dtype=np.int64)
    mapping[target] = source
    return mapping


def local_packets(h, w, phase):
    dy, dx = ((0, 0), (-4, -4), (-4, 0), (0, -4))[phase]
    gx, gy = (h - dy + 7) // 8, (w - dx + 7) // 8
    plan = np.full((gy, gx, 8, 4, 2, 32), -1, np.int32)
    members = np.full((gy, gx, 64), -1, np.int32)
    for cy in range(gy):
        for cx in range(gx):
            for t in range(4):
                y4, x4 = 2 * cx + dy // 4 + (t & 1), 2 * cy + dx // 4 + t // 2
                if not (0 <= y4 < h // 4 and 0 <= x4 < w // 4):
                    continue
                cell = (h // 4) * x4 + y4
                for p in range(8):
                    for k in range(2):
                        plan[cy, cx, p, t,
                             k] = cell * 8192 + 1024 * p + 512 * k + 16 * np.arange(32)
                for r in range(16):
                    slot = ((r >> 3) & 1) + 2 * (r & 3) + 8 * ((r >> 2) & 1)
                    members[cy, cx, 16 * t + slot] = (4 * y4 + r // 4) * w + 4 * x4 + r % 4
    return plan, members, (gx, gy, 4)


def packets(mapping, rows, channels):
    mapping = np.asarray(mapping).reshape(rows, channels)
    sources, routes = [], []
    for m in range(0, rows, 128):
        for k in range(0, channels, 64):
            tile = np.full((128, 64), -1, np.int64)
            tile[:min(128, rows - m)] = mapping[m:m + 128, k:k + 64]
            flat = tile.reshape(-1)
            valid = flat >= 0
            bases = np.unique(flat[valid] // 16)
            if len(bases) > 1024:
                raise ValueError('raw packet capacity')
            src = np.full(1024, -1, np.int32)
            src[:len(bases)] = bases * 16
            route = np.zeros(8192, np.uint16)
            route[valid] = np.searchsorted(bases, flat[valid] // 16) * 16 + flat[valid] % 16
            assert np.array_equal(src[route[valid] // 16] + route[valid] % 16, flat[valid])
            sources.append(src)
            routes.append(route)
    return np.stack(sources), np.stack(routes)
