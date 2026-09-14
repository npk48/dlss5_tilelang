"""Development-only faithful ONNX export and native/reference validation.

No Python is used by the delivered guides library. VDA exports specialize the
network spatial shape because the reference DINO positional interpolation uses
Python float/int scale factors. Source image sizes remain native/dynamic.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reference'))
import numpy as np
import torch
from torch import nn
from whitebox_pipeline.estimators import checked_models, depth_network_size, RaftSmall, MetricVideoDepth


class FlowGraph(nn.Module):
    def __init__(self, network, updates):
        super().__init__()
        self.network, self.updates = network, updates

    def forward(self, current, previous):
        return self.network(current, previous, num_flow_updates=self.updates)[-1]


class DepthGraph(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, rgb, *cache):
        depth, new = self.network(rgb, list(cache) if cache else None)
        return (depth,) + tuple(new)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(args):
    import onnx
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    models, provenance = checked_models()
    records = []
    if args.component in ('all', 'flow'):
        net = RaftSmall(models['raft-small-C_T_V2'][0], device='cpu', updates=args.flow_updates).network
        graph = FlowGraph(net, args.flow_updates).eval()
        x = torch.zeros(1, 3, 128, 160)
        path = dest / f'raft_small_u{args.flow_updates}.onnx'
        torch.onnx.export(graph, (x, x), str(path), opset_version=17,
                          input_names=['current', 'previous'], output_names=['flow'],
                          dynamic_axes={name: {2: 'height', 3: 'width'} for name in ['current', 'previous', 'flow']},
                          do_constant_folding=True)
        onnx.checker.check_model(str(path))
        records.append({'file': path.name, 'sha256': sha(path), 'dynamic_spatial': True,
                        'updates': args.flow_updates, 'minimum_hw': [128, 128], 'multiple': 8})
        print('EXPORTED', path, flush=True)
    if args.component in ('all', 'depth'):
        h, w = depth_network_size(args.height, args.width, args.depth_input_size)
        model = MetricVideoDepth(models['metric-video-depth-anything-small'][0], device='cpu', fp32=True)
        graph = DepthGraph(model.network).eval()
        x = torch.zeros(1, 3, h, w)
        with torch.inference_mode():
            initial = graph(x)
        cache = tuple(v.repeat(1, 31, 1) for v in initial[1:])
        outputs = ['depth'] + [f'new_cache_{i}' for i in range(len(cache))]
        for label, inputs in [('init', (x,)), ('step', (x,) + cache)]:
            path = dest / f'vda_small_{h}x{w}_{label}.onnx'
            names = ['rgb'] + ([f'cache_{i}' for i in range(len(cache))] if label == 'step' else [])
            torch.onnx.export(graph, inputs, str(path), opset_version=17,
                              input_names=names, output_names=outputs, do_constant_folding=True)
            onnx.checker.check_model(str(path))
            records.append({'file': path.name, 'sha256': sha(path), 'network_hw': [h, w],
                            'cache_shapes': [list(v.shape) for v in initial[1:]], 'precision': 'float32'})
            print('EXPORTED', path, flush=True)
    manifest = dest / 'native_guides.json'
    old = json.loads(manifest.read_text())['exports'] if manifest.exists() else []
    keys = {r['file'] for r in records}
    manifest.write_text(json.dumps({'format': 1, 'reference': provenance,
        'exports': [r for r in old if r['file'] not in keys] + records,
        'variant': 'Metric Small experimental causal streaming, not offline VDA',
        'precision': 'FP32; compare MetricVideoDepth(fp32=True)',
        'torch': torch.__version__, 'onnx': onnx.__version__}, indent=2), encoding='utf-8')


def frame(h, w, index):
    y, x = np.mgrid[:h, :w].astype(np.float32)
    # Analytic, reproducible translating texture with broad frequency content.
    xx = x - index * 1.75
    rgb = np.stack([.5 + .24*np.sin(xx*.13)+.21*np.cos(y*.17),
                    .5 + .24*np.sin(xx*.071+y*.099)+.18*np.cos(xx*.21-y*.14),
                    .5 + .25*np.cos(xx*.11)*np.sin(y*.19)], axis=-1)
    return np.ascontiguousarray(np.clip(rgb, 0, 1), dtype=np.float32)


def validate(args):
    dest = args.output
    work = dest / f'validation_{args.height}x{args.width}_d{args.depth_input_size}'
    work.mkdir(parents=True, exist_ok=True)
    models, _ = checked_models()
    flow = RaftSmall(models['raft-small-C_T_V2'][0], updates=args.flow_updates,
                     longest_side=args.flow_longest_side) if args.component in ('all', 'flow') else None
    depth = MetricVideoDepth(models['metric-video-depth-anything-small'][0],
                             input_size=args.depth_input_size, fp32=True) if args.component in ('all', 'depth') else None
    previous = None
    with torch.inference_mode():
        for i in range(args.frames):
            image = frame(args.height, args.width, i)
            image.tofile(work / f'rgb_{i}.f32')
            rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
            if flow and previous is not None:
                output, _ = flow(rgb, previous)
                output[0].permute(1, 2, 0).contiguous().cpu().numpy().tofile(work / f'flow_ref_{i}.f32')
            if depth:
                output, state, _ = depth.propose(rgb)
                depth.state = state
                output[0].permute(1, 2, 0).contiguous().cpu().numpy().tofile(work / f'depth_ref_{i}.f32')
            previous = rgb
            print('REFERENCE frame', i, flush=True)
    del flow, depth, previous
    torch.cuda.empty_cache()
    if not args.native_exe:
        print('Reference files only; native validation NOT run:', work)
        return
    command = [str(args.native_exe.resolve()), str(dest.resolve()), str(work.resolve()),
               str(args.height), str(args.width), str(args.frames), str(args.flow_updates),
               str(args.flow_longest_side), str(args.depth_input_size), args.component]
    env = os.environ.copy()
    if os.name == 'nt':
        # Native delivery must resolve GPU libraries beside the EXE, not from
        # Python, Torch, a development venv, or a CUDA Toolkit PATH entry.
        root = env.get('SystemRoot', r'C:\Windows')
        env['PATH'] = str(Path(root) / 'System32') + os.pathsep + root
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    (work / 'native.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    print(proc.stdout)
    if proc.returncode:
        print(proc.stderr)
        raise RuntimeError(f'Native executable failed ({proc.returncode}); see {work / "native.log"}')
    results = []
    for kind, start in [('flow', 1), ('depth', 0)]:
        if args.component not in ('all', kind):
            continue
        for i in range(start, args.frames):
            ref = np.fromfile(work / f'{kind}_ref_{i}.f32', dtype=np.float32)
            got = np.fromfile(work / f'{kind}_native_{i}.f32', dtype=np.float32)
            if got.shape != ref.shape or not np.isfinite(got).all():
                raise AssertionError(f'{kind} frame {i}: invalid output')
            error = np.abs(ref - got)
            # Fixed before native runs; tight absolute+relative tolerance, no rescaling/alignment.
            atol, rtol = (.03, .005) if kind == 'flow' else (.03, .005)
            passed = bool(np.all(error <= atol + rtol * np.abs(ref)))
            results.append({'component': kind, 'frame': i, 'mae': float(error.mean()),
                            'max_abs': float(error.max()), 'reference_mean': float(ref.mean()),
                            'atol': atol, 'rtol': rtol, 'pass': passed})
    report = {'source_hw': [args.height, args.width], 'depth_network_hw': depth_network_size(args.height, args.width, args.depth_input_size),
              'frames': args.frames, 'flow_updates': args.flow_updates, 'component': args.component,
              'reference_precision': 'fp32', 'native_stdout': proc.stdout,
              'results': results, 'pass': all(r['pass'] for r in results)}
    (work / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise AssertionError('Native/reference tolerance failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'validate'])
    parser.add_argument('--output', type=Path, default=ROOT / 'model/native_guides')
    parser.add_argument('--component', choices=['all', 'flow', 'depth'], default='all')
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--flow-updates', type=int, default=8)
    parser.add_argument('--flow-longest-side', type=int, default=512)
    parser.add_argument('--depth-input-size', type=int, default=518)
    parser.add_argument('--frames', type=int, default=13)
    parser.add_argument('--native-exe', type=Path)
    args = parser.parse_args()
    if min(args.height, args.width, args.frames, args.flow_updates) <= 0 or args.depth_input_size < 28 or args.flow_longest_side < 128:
        parser.error('invalid dimensions/counts/work sizes')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        (export if args.action == 'export' else validate)(args)


if __name__ == '__main__':
    main()
