"""Run the CUDA NR reference on a prepared packet or deterministic demo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from runtime import bootstrap
    from cuda_nr.api import load_runtime, model_module
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=bootstrap.MODEL / 'weights_ht_blob.bin')
    parser.add_argument('--toolchain', type=Path, help='CUDA 12.8 root (nvrtc/runtime/cccl)')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='Float32 Bx16xHxW .npy')
    inputs.add_argument('--demo-size', type=int, nargs=2, metavar=('H', 'W'))
    parser.add_argument('--output', type=Path, required=True, help='Float32 BxHxWx4 .npy')
    args = parser.parse_args()
    if args.toolchain:
        os.environ['NATIVE_NR_TOOLCHAIN'] = str(args.toolchain.resolve())
    torch.set_grad_enabled(False)
    runtime, _, load_report = load_runtime(args.weights, return_details=True)
    try:
        if args.input:
            packet = torch.from_numpy(np.load(args.input, allow_pickle=False)).to('cuda')
        else:
            h, w = args.demo_size
            y = torch.arange(h, device='cuda', dtype=torch.float32)[:, None] / max(h - 1, 1)
            x = torch.arange(w, device='cuda', dtype=torch.float32)[None, :] / max(w - 1, 1)
            color = torch.stack((x.expand(h, w), y.expand(h, w), ((x + y) / 2).expand(h, w)), -1)[None]
            motion = torch.zeros((1, h, w, 2), device='cuda')
            motion[..., 0], motion[..., 1] = 0.25, -0.5
            frame = model_module.FrameInputs(
                color=color, prev_output=color.roll(3, dims=2), mvec=motion,
                mvec_scale_xy=torch.ones(2, device='cuda'),
                output_dimensions_wh=torch.tensor([w, h]), style=1,
                local_structure_strength=2, local_tone_strength=1.2, use_auto_mask=False)
            packet = model_module.build_preblock_features(frame, frame=37).contiguous()
        head = runtime.infer_minimal(packet).cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, head, allow_pickle=False)
        print(json.dumps({'shape': list(head.shape), 'dtype': str(head.dtype),
                          'sha256': hashlib.sha256(head.tobytes()).hexdigest(),
                          'model_loader': load_report, 'runtime': runtime.report()}, indent=2))
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
