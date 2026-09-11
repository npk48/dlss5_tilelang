from .storage import Storage, U, I, Weights
from .layouts import Shape, align, dep, raw16, encoder_pixels, pointmap, halfmap, repack99, local_packets, packets
from .source import prepare, DEEP_UNITS

# outer_runtime.py
"""Live native block0..22 / block48..70 packet executor.

All saved skips and auxiliary Half rails have independent owned storage. The
middle region is supplied only by DeepRuntime, never by a Torch/TileLang fallback.
"""
import ctypes as C
from pathlib import Path
import time

import numpy as np
import torch


class Cross(C.Structure):
    _fields_ = [
        (n, U) for n in ('pool_input', 'pool_output', 'up_inverse', 'projection', 'gate', 'matrix')
    ] + [(n, I) for n in ('pool_out', 'skip')]


class Layout(C.Structure):
    _fields_ = [(n, I) for n in ('height', 'width', 'grid_x', 'shift_x', 'shift_y', 'flags')]


class OuterRuntime:

    def __init__(self, model, height, width, *, device=None, weights_path=None):
        tick = time.perf_counter()
        self.shape = Shape(height, width)
        self.device = torch.device(device or next(model.buffers()).device)
        self.storage = Storage(self.device)
        self.closed = False
        self.calls = 0
        self.encoder_steps, self.decoder_steps = [], []
        self.prepare_seconds = 0.0
        try:
            self.weights = Weights(model, weights_path, self.storage)
            self._prepare(model)
            self.prepare_seconds = time.perf_counter() - tick - self.storage.compile_seconds
        except BaseException:
            self.close()
            raise

    def _prepare(self, model):
        shape, d, wt = self.shape, self.storage, self.weights
        h, w = shape.height, shape.width
        ds, up = prepare(shape)
        defines = {'NR_H': h, 'NR_W': w}
        units = {}

        def logical_unit(name):
            if name not in units:
                units[name] = d.unit(name, defines)
            return units[name]

        offsets, lengths, cursor = {}, {}, 0

        def reserve(name, size):
            nonlocal cursor
            cursor = (cursor + 255) // 256 * 256
            offsets[name], lengths[name] = cursor, int(size)
            cursor += int(size)

        reserve('skip0', h * w * 32)
        reserve('out0', h * w // 4 * 32)
        self.plans = {b: shape.outer(b) for b in (*range(1, 23), *range(48, 70))}
        for b, p in self.plans.items():
            hh, ww = p['shape']
            heads = p['heads']
            padded_w = (ww + 7) // 8 * 8 if heads == 8 and b != 55 else ww
            reserve(f'out{b}', hh * padded_w * heads * 32)
            if b in (4, 8, 14, 22):
                ph, pw = p['pool_shape']
                reserve(f'pool{b}', ph * pw * heads * 64)
        deep_h, deep_w = shape.deep_hw
        reserve('deep_out', deep_h * deep_w * 512)
        cursor = (cursor + 255) // 256 * 256
        counter_begin = cursor
        for b, p in self.plans.items():
            reserve(f'counter{b}', p['grid'][0] * p['grid'][1] * 4)
        if cursor >= 2**31:
            raise ValueError('Native outer arena exceeds signed 32-bit descriptor offsets')
        self.arena = d.tensor(cursor)
        base = self.arena.data_ptr()
        self.offsets, self.lengths = offsets, lengths
        self.deep_input = self.arena[offsets['pool22']:offsets['pool22'] + lengths['pool22']]
        self.deep_output = self.arena[offsets['deep_out']:offsets['deep_out'] + lengths['deep_out']]
        self.head = d.tensor((1, h, w, 4), torch.float32)
        self.status = d.tensor(1, torch.int32)
        util = logical_unit('utility.cu')
        n = (cursor - counter_begin) // 4
        self.clear = d.step(util, 'clear_counters', ((n + 255) // 256, 1, 1), (256, 1, 1), 0,
                            [U(base + counter_begin),
                             I(n), U(self.status.data_ptr())])
        # Retain the actual auxiliary rails, including writes not needed by the
        # next arithmetic kernel. No release-aliasing of native publications.
        self.aux = {
            'DS_pre': d.tensor(h * w // 4 * 32, torch.float16),
            'UP_mixed': d.tensor(h * w // 4 * 32, torch.float16),
            'UP147_projection': d.tensor(h * w // 64 * 64, torch.float16),
            'UP141_projection': d.tensor(h * w // 256 * 128, torch.float16),
            'UP133_projection': d.tensor(deep_h * deep_w * 256, torch.float16),
        }
        self.input_argument = U(0)
        self.pre = d.step(logical_unit('shallow_static_geometry/one.cu'), 'block0_native_packet',
                          (w // 8, h // 8, 1), (32, 1, 1), 0, [
                              wt.one(0), self.input_argument,
                              U(wt.input_adapter(model)),
                              U(base + offsets['skip0']),
                              U(base + offsets['out0']),
                              U(self.status.data_ptr())
                          ])
        previous = offsets['out0']
        for b in (*range(1, 23), *range(48, 70)):
            p = self.plans[b]
            heads = p['heads']
            seq = p['seq']
            hh, ww = p['shape']
            gx, gy, _ = p['grid']
            if b == 48:
                previous = offsets['deep_out']
            out, counter = offsets[f'out{b}'], offsets[f'counter{b}']
            steps = self.encoder_steps if b <= 22 else self.decoder_steps
            geometry = ((gx * gy, 1, 1), (32, heads, 1))
            cross = Cross()
            if heads == 1:
                values = [
                    U(base),
                    wt.one(b),
                    U(0),
                    U(0),
                    I(previous),
                    I(out),
                    I(counter),
                    U(0),
                    U(0)
                ]
                unit, name = 'shallow_static_geometry/one.cu', f'outer1_static_{seq}'
                if b == 4:
                    unit, name = 'shallow_static_geometry/transition.cu', 'outer1_static_ds'
                    values[8] = U(self.aux['DS_pre'].data_ptr())
                    values += [U(wt.ptr(b) + 0x50b0), U(0), U(0), U(0), I(offsets['pool4'])]
                if b == 66:
                    unit, name = 'shallow_static_geometry/transition.cu', 'outer1_static_up'
                    gate = np.frombuffer(wt.raw(66), '<f2', 32, 0x2860)
                    order = [
                        0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20,
                        21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31
                    ]
                    values[7] = U(self.aux['UP_mixed'].data_ptr())
                    values += [
                        U(wt.ptr(66) + 0x2000),
                        U(0),
                        U(0),
                        U(0),
                        U(0),
                        U(d.upload(gate[order])),
                        I(offsets['out4'])
                    ]
                steps.append(d.step(logical_unit(unit), name, *geometry, 0, values))
            else:
                if b in (8, 14, 22):
                    pi, po = ds[heads]
                    cross.pool_input, cross.pool_output = d.upload(pi), d.upload(po)
                    cross.pool_out = offsets[f'pool{b}']
                    cross.matrix = wt.ptr(b) + {2: 0xf130, 4: 0x30230, 8: 0xa8440}[heads]
                if b in (48, 56, 62):
                    ui, inverse, _ = up[heads]
                    cross.up_inverse = d.upload(inverse)
                    cross.projection = self.aux[f'UP{seq}_projection'].data_ptr()
                    cross.skip = offsets[f'out{ {48:22,56:14,62:8}[b] }']
                    cross.gate = wt.ptr(b) + {2: 0x9080, 4: 0x20100, 8: 0x78200}[heads]
                    ui_ptr = d.upload(ui)
                    if b == 62:
                        steps.append(
                            d.step(logical_unit('wide_transition_packet/two.cu'), 'streamed_project2',
                                   ((len(ui) + 15) // 16, 1, 1), (32, 2, 1), 0, [
                                       U(base),
                                       U(cross.projection),
                                       U(wt.ptr(b) + 0x7000),
                                       U(ui_ptr),
                                       I(previous),
                                       I(len(ui))
                                   ]))
                else:
                    ui_ptr = 0
                if heads == 2:
                    meta = d.upload(np.array([hh, ww, gx, *p['shift'], seq], dtype='<i4'))
                    unit, name = ('wide_transition_packet/two.cu',
                                  'packet2_ds') if b == 8 else ('two_physical_seed/two.cu',
                                                                'streamed2')
                    values = [
                        U(base),
                        wt.two(b),
                        U(meta),
                        U(0),
                        I(previous),
                        I(out),
                        I(counter), cross
                    ]
                    shared = 12288 if b == 8 else 4096
                else:
                    flags = 1 if b in (9, 15) else 2 if b in (55, 61) else 0
                    layout = Layout(hh, ww, gx, *p['shift'], flags)
                    values = [
                        U(base),
                        U(wt.ptr(b)), layout,
                        I(previous),
                        I(out),
                        I(counter),
                        I(b in (48, 56)),
                        U(base), cross
                    ]
                    if b in (14, 22):
                        unit, name = f'wide_transition_packet/ds{heads}.cu', f'packet{heads}_ds'
                        values += [U(0)]
                    elif b in (48, 56):
                        unit, name = f'wide_transition_packet/up{heads}.cu', f'packet{heads}_up'
                        values += [U(ui_ptr)]
                    elif heads == 4:
                        unit = 'wide_input_packet/wide4.cu'
                        name = 'four_input' if flags == 1 else 'four_output' if flags == 2 else 'four_ordinary'
                    else:
                        unit = 'eight_full_projection/eight.cu'
                        name = 'full8_inpview' if flags == 1 else 'full8_outview' if flags == 2 else 'full8_chained'
                    shared = heads * 2048
                steps.append(d.step(logical_unit(unit), name, *geometry, shared, values))
            previous = offsets[f'pool{b}'] if b in (4, 8, 14, 22) else out
            if b == 22:
                ph, pw = p['pool_shape']
                n = ph * pw * 512
                steps.append(
                    d.step(util, 'pool8_padding', ((n + 255) // 256, 1, 1), (256, 1, 1), 0,
                           [U(base), I(previous), I(hh),
                            I(ww), I(ph), I(pw)]))
        mg, sg, readout = wt.post(model)
        self.post = d.step(logical_unit('post_static_packet/post.cu'), 'post70_native_head',
                           (w // 8 + 1, h // 8 + 1, 1), (32, 1, 1), 0, [
                               wt.one(70),
                               U(base + previous),
                               U(base + offsets['skip0']),
                               U(mg),
                               U(sg),
                               U(readout),
                               U(self.head.data_ptr()),
                               U(self.status.data_ptr())
                           ])

    def encode(self, prepared_features):
        self.storage.guard()
        self.input_argument.value = prepared_features.data_ptr()
        self.clear()
        self.pre()
        for step in self.encoder_steps:
            step()
        return self.deep_input

    def decode(self, deep_output):
        if deep_output.dtype != torch.uint8 or deep_output.device != self.device or not deep_output.is_contiguous(
        ) or deep_output.shape != self.deep_output.shape:
            raise ValueError('DeepRuntime output does not match the native out132 byte layout')
        self.deep_output.copy_(deep_output)
        for step in self.decoder_steps:
            step()
        self.post()
        status = int(self.status.item())
        if status:
            raise FloatingPointError(
                f'Native NR packet/head validation failed (1=nonfinite or overflowing Half input, 4=nonfinite head): {status}'
            )
        self.calls += 1
        return self.head

    def describe(self):
        return {
            **self.shape.describe(), 'calls': self.calls,
            'outer_blocks': 46,
            'input_layout': 'BCHW16 Float32 packet, original InputBlock Half cast',
            'output_layout': 'BHWC4 Float32 Half-valued learned head',
            'deep_input_bytes': self.deep_input.numel(),
            'deep_output_bytes': self.deep_output.numel(),
            'arena_bytes': self.arena.numel(),
            'aux_bytes': {
                k: v.numel() * v.element_size()
                for k, v in self.aux.items()
            },
            'encoder_launches': len(self.encoder_steps) + 1,
            'decoder_launches': len(self.decoder_steps) + 1,
            'prepare_seconds': self.prepare_seconds,
            'compile_seconds': self.storage.compile_seconds
        }

    def close(self):
        if self.closed:
            return
        self.storage.close()
        self.closed = True


# deep_runtime.py
"""Independent raw-input/raw-output native blocks23..47.

Preparation reads only frozen BIN weights and static CUDA/layout sources. Every
inference is a sequence of ordinary default-stream launches with retained typed
arguments; no capture replay, tensor arithmetic backend, or snapshot restoration.
"""
import ctypes as C
import hashlib
from pathlib import Path
import struct

import numpy as np
import torch



class CompletionRegion(C.Structure):
    _fields_ = [('pointer', U), ('count', I), ('value', I)]


class Completion(C.Structure):
    _fields_ = [('regions', CompletionRegion * 2), ('count', I), ('padding', I)]


def _weights(path):
    blob = Path(path).read_bytes()
    if len(blob) < 8 or struct.unpack_from('<Q', blob)[0] != len(blob):
        raise ValueError('Incomplete frozen weights BIN')
    cursor, records = 8, {}
    while cursor < len(blob):
        if cursor + 8 > len(blob):
            raise ValueError('Truncated BIN record')
        length = struct.unpack_from('<Q', blob, cursor)[0]
        cursor += 8
        if not 1 <= length <= 4096 or cursor + length + 8 > len(blob):
            raise ValueError('Invalid BIN name')
        name = blob[cursor:cursor + length].decode('ascii')
        cursor += length
        span = struct.unpack_from('<Q', blob, cursor)[0]
        body = cursor + 8
        cursor = body + span
        if span < 40 or cursor > len(blob):
            raise ValueError('Invalid BIN body')
        total, size = struct.unpack_from('<QQ', blob, body)
        start, end = body + 20, body + 20 + size
        if total != span or size % 2 or end + 16 > cursor:
            raise ValueError('Invalid BIN payload')
        rank = struct.unpack_from('<Q', blob, end + 8)[0]
        if end + 16 + 4 * rank != cursor or name in records:
            raise ValueError('Invalid BIN trailer')
        records[name] = blob[start:end]
    if len(records) != 153:
        raise ValueError('Expected 153 frozen weight records')
    return records


class DeepRuntime:

    def __init__(self, model, logical_h, logical_w, *, device=None, weights_path=None):
        h, w = int(logical_h), int(logical_w)
        # Product block22 emits complete physical4 cells, including ragged8 cells.
        if h != logical_h or w != logical_w or h <= 0 or w <= 0 or h % 4 or w % 4:
            raise ValueError('Deep H/W are positive block22 physical dimensions, multiples of four')
        self.h, self.w = h, w
        self.ph, self.pw = align(h, 8) // 2, align(w, 8) // 2
        self.tokens, self.rows = self.ph * self.pw, h * w
        if max(self.rows * 512, align(self.tokens, 128) * 4096 * 8) > 2**31 - 1:
            raise ValueError('Deep raw addressing exceeds signed int32')
        self.device = torch.device(device if device is not None else 'cuda')
        self.closed = False
        self.steps = []
        self.launches = []
        self.aux = []
        self.stage_outputs = {}
        self._completion_refs = []
        self.storage = None
        self.input_bytes = self.output_bytes = self.rows * 512
        self.runtime_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.defines = dict(DH=h,
                            DW=w,
                            SH=(h + 7) // 8,
                            PW=self.pw,
                            PH=self.ph,
                            DT=self.tokens,
                            DP=align(self.tokens, 128),
                            MT=(self.tokens + 127) // 128)
        records = _weights(weights_path if weights_path is not None else model.weights_path)
        try:
            with torch.cuda.device(self.device):
                self.storage = Storage(self.device)
                self.device = self.storage.device
                self.storage.guard()
                self.modules = {
                    name: self.storage.unit(DEEP_UNITS[name], self.defines)
                    for name in ('sixteen', 'local', 'matrix', 'qkv', 'attention', 'transition',
                                 'repack', 'split')
                }
                self.weight_ptrs = {}
                for b in range(23, 48):
                    layers = range(5) if b in range(30, 39) else range(1) if b == 39 else range(4)
                    for layer in layers:
                        if b in range(31, 39) and layer == 3:
                            continue
                        data = records[f'block{b}.layer{layer}.layer']
                        expected = ((0x80000, 0x40400, 0xe0040, 0x40400, 0x80010)[layer]
                                    if b <= 30 or b >= 40 else 0x80400 if b == 39 else
                                    (0x400010, 0x400800, 0x300080, 2, 0x100800)[layer])
                        if len(data) != expected:
                            raise ValueError(f'Unexpected frozen block{b}.layer{layer} weight size')
                        self.weight_ptrs[b, layer] = self.storage.upload(data)
                self.output = self.storage.tensor(self.output_bytes)
                self.input_arg = U(0)
                self._prepare()
                self.aux_slab = self.storage.tensor(sum(r['count'] for r in self.aux), torch.int32)
                offset = 0
                for completion, index, region in self._completion_refs:
                    region['pointer'] = self.aux_slab.data_ptr() + 4 * offset
                    region['offset'] = offset
                    completion.regions[index].pointer = region['pointer']
                    offset += region['count']
        except BaseException:
            self.close()
            raise

    def _buffer(self, count, dtype=torch.uint8):
        # Zero initialization is once-only for the padding beyond real M. All real
        # output/partial bytes are rewritten by their producer on every frame.
        value = self.storage.tensor(count, dtype)
        value.zero_()
        return value.data_ptr()

    def _map(self, array):
        return self.storage.upload(np.asarray(array, np.int32))

    def _completion(self, count, split=False, split_value=3, first_only=False):
        result = Completion()
        result.count = 2 if split and not first_only else 1
        for i in range(result.count):
            value = split_value if split and i == 0 else 0
            result.regions[i] = CompletionRegion(0, count, value)
            region = dict(pointer=0, count=count, value=value)
            self.aux.append(region)
            # Step retains these ctypes objects; patch addresses after slab allocation.
            self._completion_refs.append((result, i, region))
        return result

    def _add(self, seq, unit, name, grid, block, values):
        self.steps.append(self.storage.step(self.modules[unit], name, grid, block, 0, values))
        self.launches.append(
            dict(sequence=seq, name=name, unit=unit, grid=list(grid), block=list(block)))

    def _flat(self, seq, unit, name, count, values):
        self._add(seq, unit, name, ((count + 255) // 256, 1, 1), (256, 1, 1), values)

    def _repack(self, seq, source, target, imap, omap, count, completion=False):
        self._flat(seq, 'repack', 'raw_repack_completion', count, [
            U(source),
            U(target),
            U(imap),
            U(omap),
            I(count),
            self._completion(int(completion)) if completion else Completion()
        ])

    def _prepare(self):
        h, w, t, dp = self.h, self.w, self.tokens, self.defines['DP']
        sh, sw = (h + 7) // 8, (w + 7) // 8
        enc, dec = raw16(h, w), raw16(h, w, decoder=True)
        pc, pi = pointmap(self.ph, self.pw, 1024), pointmap(self.ph, self.pw, 1024, True)
        ff = pointmap(self.ph, self.pw, 4096)
        pool = raw16(self.ph, self.pw, pool=True)
        terminal = raw16(self.ph, self.pw, pool=True, channels=1024)
        ni = np.arange(1024, dtype=np.int64)
        ri, rc = dep(ni, (0, 1, 3, 4, 2, 5, 6, 7, 8, 9)), dep(ni, (0, 3, 4, 1, 2, 5, 6, 7, 8, 9))
        nextroute = np.argsort(rc)[ri]
        nextids = (np.arange(t)[:, None] * 1024 + nextroute[None, :]).reshape(-1)
        weights = self.weight_ptrs
        # Raw16 scratch ping-pongs; long-lived skip56 is separate from decoder.
        a, b, att, c = [self._buffer(self.rows * 512) for _ in range(4)]
        poolraw = self._buffer(t * 512)
        skip56 = self._buffer(self.rows * 512)
        plans = {}
        for phase in range(4):
            plan, members, grid = local_packets(h, w, phase)
            plans[phase] = (self._map(plan), self._map(members), grid)

        def swin(block, current):
            seq = 25 + (block - 23) * 4 if block < 31 else 101 + (block - 40) * 4
            decoder = int(block >= 40)
            first = block == 23
            self._add(seq, 'sixteen', 'first25_completion' if first else 'chained_completion',
                      (sh, sw, 2), (32, 4 if first else 8, 1), [
                          current if first else U(current),
                          U(weights[block, 0]),
                          U(0),
                          U(a),
                          I(decoder),
                          self._completion(sh * sw * 2)
                      ])
            self._add(seq + 1, 'sixteen',
                      'special26_completion' if first else 'project4_completion', (sh * 2, sw, 1),
                      (32, 4, 1), [
                          U(a),
                          U(weights[block, 1]), current if first else U(current),
                          U(0),
                          U(0),
                          U(b),
                          U(0),
                          U(0),
                          I(decoder),
                          self._completion(sh * sw * 2)
                      ])
            plan, members, grid = plans[(block - (40 if decoder else 23)) % 4]
            self._add(seq + 2, 'local', 'joint_local64_completion', grid, (32, 4, 1), [
                U(b),
                U(plan),
                U(weights[block, 2]),
                U(0),
                U(att),
                U(0),
                U(members),
                self._completion(np.prod(grid).item())
            ])
            name = 'pool56' if block == 30 else 'out132' if block == 47 else 'project8_completion'
            out = self.output.data_ptr() if block == 47 else c
            values = [
                U(att),
                U(weights[block, 3]),
                U(b),
                U(0),
                U(0),
                U(out),
                U(0),
                U(poolraw if block == 30 else 0),
                I(decoder)
            ]
            if name.endswith('_completion'):
                values.append(self._completion(sh * sw * 2))
            self._add(seq + 3, 'sixteen', name, (sh * 2, sw, 1),
                      (32, 4 if block in (30, 47) else 8, 1), values)
            self.stage_outputs[seq + 3] = (out, self.rows * 512)
            return out

        current = self.input_arg
        for block in range(23, 31):
            current = swin(block, current)
        self._repack(56, current, skip56, self._map(enc), 0, self.rows * 512, sh * sw * 2)
        x0, x1, q, k, v, at = [self._buffer(dp * 1024) for _ in range(6)]
        hidden = self._buffer(dp * 4096)
        partial = self._buffer(t * 1024 * 4, torch.float16)
        scratch = self._buffer(dp * 1024 * 3, torch.float16)
        hp = self._map(halfmap(pc))
        pcm = self._map(pc)

        def matrix_args(source,
                        weight,
                        rows,
                        K,
                        N,
                        mode,
                        splits,
                        raw=0,
                        outmap=0,
                        part=0,
                        skip=0,
                        gate=0,
                        source_map=None):
            pkt, route = (0, 0) if source_map is None else tuple(
                self.storage.upload(x) for x in packets(source_map, rows, K))
            return [
                U(source),
                U(weight),
                U(skip),
                U(0),
                U(0),
                U(part),
                I(rows),
                I(K),
                I(N),
                I(mode),
                I(splits),
                I(gate),
                I(0),
                U(0),
                I(0),
                U(raw),
                U(outmap),
                U(0),
                U(0),
                U(0),
                U(0),
                U(pkt),
                U(route),
                U(0)
            ]

        self._add(
            57, 'transition', 'matrix_phase', ((t + 127) // 128, 8, 1), (128, 1, 1),
            matrix_args(poolraw,
                        weights[30, 4],
                        t,
                        512,
                        1024,
                        1,
                        1,
                        x0,
                        self._map(terminal),
                        source_map=pool))
        self._repack(58, x0, x1, self._map(terminal[nextids]), self._map(pi), t * 1024)
        current = x1
        for block in range(31, 39):
            seq = 59 + (block - 31) * 5
            mt = (t + 127) // 128
            args = matrix_args(current, weights[block, 0], t, 1024, 4096, 0, 1, hidden,
                               self._map(ff))
            self._add(seq, 'matrix', 'matrix_expand_completion', (mt * 32, 1, 1), (32, 4, 1),
                      args + [I(64), self._completion(mt * 32)])
            args = matrix_args(hidden,
                               weights[block, 1],
                               t,
                               4096,
                               1024,
                               2,
                               4,
                               part=partial,
                               skip=current,
                               gate=0x400000)
            self._add(seq + 1, 'matrix', 'matrix_contract', (mt * 8, 1, 4), (32, 4, 1),
                      args + [I(64)])
            self._flat(seq + 1, 'split', 'publish_split', t * 1024,
                       [U(partial), U(scratch), U(hp),
                        I(t * 1024), I(4)])
            self._flat(seq + 1, 'transition', 'merge_phase_completion', t * 1024, [
                U(partial),
                U(scratch),
                U(hp),
                U(0),
                I(t * 1024),
                I(4),
                U(0),
                I(0),
                U(x0),
                U(pcm),
                self._completion(mt * 8, True)
            ])
            # QKV p0 writes the actual raw Half scratch; p1 reads and HADDs it.
            args = [U(x0), U(weights[block, 2])] + [U(0)] * 4 + [U(scratch)] + [U(0)] * 4 + [
                U(q), U(k), U(v), U(weights[block, 2]),
                I(t)
            ] + [U(0)] * 5
            self._add(seq + 2, 'qkv', 'joint_qkv_completion', (mt * 16, 1, 1), (32, 4, 1),
                      args + [self._completion(mt * 16, True, 1)])
            self._add(
                seq + 3, 'attention', 'joint_attention_completion', (32, (t + 255) // 256, 1),
                (32, 4, 1),
                [U(q), U(k), U(v), U(0),
                 U(at), U(0), U(0), I(t),
                 I(32), I(1)] + [U(0)] * 4 + [self._completion(((t + 127) // 128) * 32)])
            args = matrix_args(at,
                               weights[block, 4],
                               t,
                               1024,
                               1024,
                               2,
                               4,
                               part=partial,
                               skip=x0,
                               gate=0x100000)
            self._add(seq + 4, 'matrix', 'matrix_projection', (mt * 8, 1, 4), (32, 4, 1),
                      args + [I(32)])
            self._flat(seq + 4, 'split', 'publish_split', t * 1024,
                       [U(partial), U(scratch), U(hp),
                        I(t * 1024), I(4)])
            pm = pc if block == 38 else pi[np.argsort(nextids)]
            self._flat(seq + 4, 'transition', 'merge_phase_completion', t * 1024, [
                U(partial),
                U(scratch),
                U(hp),
                U(0),
                I(t * 1024),
                I(4),
                U(0),
                I(0),
                U(x1),
                U(self._map(pm)),
                self._completion(mt * 8, True, first_only=block == 38)
            ])
            current = x1
            self.stage_outputs[seq + 4] = (current, t * 1024)
        r99 = repack99(self.ph, self.pw)
        self._flat(99, 'split', 'permute', t * 1024,
                   [U(current), U(x0), U(self._map(r99)),
                    I(t * 1024)])
        up_half = self._buffer(t * 512, torch.float16)
        upmap = self._map(halfmap(pool))
        self._add(
            100, 'transition', 'matrix_phase', ((t + 127) // 128, 4, 4), (128, 1, 1),
            matrix_args(x0,
                        weights[39, 0],
                        t,
                        1024,
                        512,
                        2,
                        4,
                        part=partial,
                        source_map=np.argsort(r99)[pc]))
        self._flat(100, 'split', 'publish_split', t * 512,
                   [U(partial), U(scratch), U(upmap),
                    I(t * 512), I(4)])
        self._flat(100, 'transition', 'merge_phase', t * 512, [
            U(partial),
            U(scratch),
            U(upmap),
            U(up_half),
            I(t * 512),
            I(4),
            U(0),
            I(0),
            U(0),
            U(0)
        ])
        ep = (encoder_pixels(h, w)[:, None] * 512 + np.arange(512)[None, :]).reshape(-1)
        self._flat(100, 'repack', 'up_exit_completion', self.rows * 512, [
            U(up_half),
            U(skip56),
            U(weights[39, 0] + 0x80000),
            U(b),
            U(self._map(ep)),
            U(self._map(dec)),
            I(self.rows * 512),
            I(w),
            I(self.pw),
            self._completion(sh * sw * 2, True)
        ])
        current = b
        for block in range(40, 48):
            current = swin(block, current)

    def infer(self, input_tensor):
        if self.closed:
            raise RuntimeError('DeepRuntime is closed')
        if (input_tensor.device != self.device or input_tensor.dtype != torch.uint8
                or input_tensor.ndim != 1 or not input_tensor.is_contiguous()
                or input_tensor.numel() != self.input_bytes):
            raise ValueError(
                f'Deep input must be contiguous CUDA uint8[{self.input_bytes}] on {self.device}')
        if input_tensor.data_ptr() % 16:
            raise ValueError('Deep raw input must be 16-byte aligned')
        if input_tensor.data_ptr() == self.output.data_ptr():
            raise ValueError('Deep input must not alias its owned output')
        with torch.cuda.device(self.device):
            self.storage.guard()
            self.input_arg.value = input_tensor.data_ptr()
            self.aux_slab.fill_(-1)
            for step in self.steps:
                step()
        self._last_input = input_tensor
        return self.output

    def describe(self):
        return dict(
            input_bytes=self.input_bytes,
            output_bytes=self.output_bytes,
            input_layout='raw16first512',
            output_layout='out132_planarN16',
            logical_h=self.h,
            logical_w=self.w,
            shape_chain=[
                dict(blocks='23..30', h=self.h, w=self.w, c=512),
                dict(blocks='30pool..38', h=self.ph, w=self.pw, c=1024, tokens=self.tokens),
                dict(blocks='39..47', h=self.h, w=self.w, c=512)
            ],
            real_new_kernel_calls=len(self.steps),
            fallback_calls=0,
            default_stream=0,
            raw_split_publication=True,
            weight_source='frozen BIN',
            kernels=self.launches,
            shape_defines=self.defines,
            counter_regions=len(self.aux),
            counter_clear_operations_per_frame=1,
            counter_clear='single int32 fill(-1) over the owned contiguous completion slab, stream0',
            runtime_source_sha256=self.runtime_sha256,
            closed=self.closed)

    def close(self):
        if self.closed:
            return
        if self.storage is not None:
            self.storage.close()
        self.steps.clear()
        self.output = None
        self._last_input = None
        self.closed = True


# runtime.py
"""Cache lifecycle for the TileLang NR plan: one prepared shape per entry."""
import time
import torch


class PlanRuntime:

    def __init__(self, model, weights_path=None, max_cached_shapes=2):
        if max_cached_shapes < 1:
            raise ValueError('At least one native shape cache entry is required')
        self.max_cached_shapes = int(max_cached_shapes)
        self.evictions = 0
        self.model = model
        self.weights_path = weights_path or model.weights_path
        self.entries = {}
        self.calls = self.samples = self.failures = 0
        self.prepare_seconds = self.compile_seconds = 0.0
        self.last_frame = None
        self.closed = False

    def _prepare(self, key, features):
        b, _, h, w = features.shape
        shape = Shape(h, w)
        outer = deep = None
        tick = time.perf_counter()
        try:
            # A persistent WebUI must not retain one full arena/weight copy for
            # every image size ever submitted. Returned heads are independently
            # owned; close synchronizes before retiring the internal workspace.
            while len(self.entries) >= self.max_cached_shapes:
                oldest = next(iter(self.entries))
                old_outer, old_deep = self.entries.pop(oldest)
                try:
                    old_deep.close()
                finally:
                    old_outer.close()
                self.evictions += 1
            outer = OuterRuntime(self.model,
                                 h,
                                 w,
                                 device=features.device,
                                 weights_path=self.weights_path)
            deep = DeepRuntime(self.model,
                               *shape.deep_hw,
                               device=features.device,
                               weights_path=self.weights_path)
            if deep.input_bytes != outer.deep_input.numel(
            ) or deep.output_bytes != outer.deep_output.numel():
                raise ValueError('Native outer/deep byte boundary mismatch')
            self.entries[key] = outer, deep
            compile_seconds = outer.storage.compile_seconds + deep.storage.compile_seconds
            self.compile_seconds += compile_seconds
            self.prepare_seconds += time.perf_counter() - tick - compile_seconds
            return self.entries[key]
        except BaseException:
            if deep is not None:
                deep.close()
            if outer is not None:
                outer.close()
            raise

    @torch.inference_mode()
    def infer_minimal(self, prepared_features, *, collect_trace=False, return_result=False):
        if self.closed:
            raise RuntimeError('Native NR is closed')
        if collect_trace or return_result:
            raise NotImplementedError(
                'Native NR minimal head inference does not yet export Torch latent/BlockTrace results; explicitly select legacy for those diagnostics'
            )
        x = prepared_features
        if x.ndim != 4 or x.shape[0] < 1 or x.shape[
                1] != 16 or x.dtype != torch.float32 or not x.is_cuda:
            raise ValueError('Native NR requires a CUDA BCHW16 Float32 prepared packet')
        if not x.is_contiguous():
            raise ValueError('Native NR requires a contiguous prepared packet')
        model_device = next(self.model.buffers()).device
        if x.device != model_device:
            raise ValueError('Native NR packet and frozen model must share a device')
        if torch.cuda.current_stream(x.device).cuda_stream != 0:
            raise RuntimeError(
                'Native NR requires default CUDA stream 0; no Graph/nondefault-stream path is qualified'
            )
        if torch.cuda.get_device_capability(x.device) != (8, 9):
            raise RuntimeError(
                'Native NR selected kernels require SM89; there is no automatic fallback')
        key = (*x.shape, x.device.index, x.dtype)
        try:
            with torch.cuda.device(x.device):
                entry = self.entries.pop(key, None)
                if entry is None:
                    entry = self._prepare(key, x)
                else:
                    self.entries[key] = entry
                outer, deep = entry
                # Public model outputs must survive subsequent calls. Workspaces
                # are cached, but the returned head has independent ownership.
                output = torch.empty((x.shape[0], x.shape[2], x.shape[3], 4),
                                     dtype=torch.float32,
                                     device=x.device)
                for batch in range(x.shape[0]):
                    encoded = outer.encode(x[batch:batch + 1])
                    middle = deep.infer(encoded)
                    output[batch:batch + 1].copy_(outer.decode(middle))
                self.calls += 1
                self.samples += x.shape[0]
                self.last_frame = {
                    'neural_hw': list(x.shape[2:]),
                    'batch': x.shape[0],
                    'blocks_per_sample': 71,
                    'nr_backend': 'cuda',
                    'outer_launches_per_sample':
                    len(outer.encoder_steps) + len(outer.decoder_steps) + 2
                }
                return output
        except BaseException:
            self.failures += 1
            entry = self.entries.pop(key, None)
            if entry is not None:
                entry[1].close()
                entry[0].close()
            raise

    def report(self):
        return {
            'calls': self.calls,
            'samples': self.samples,
            'failures': self.failures,
            'fallback_calls': 0,
            'cached_shapes': len(self.entries),
            'max_cached_shapes': self.max_cached_shapes,
            'evictions': self.evictions,
            'prepare_seconds': self.prepare_seconds,
            'compile_seconds': self.compile_seconds,
            'last_frame': self.last_frame,
            'trace_supported': False,
            'shape_policy': 'original prepared packet; no fixed-resolution resizing or fallback'
        }

    def close(self):
        if self.closed:
            return
        try:
            for outer, deep in self.entries.values():
                try:
                    deep.close()
                finally:
                    outer.close()
        finally:
            self.entries.clear()
            self.closed = True
