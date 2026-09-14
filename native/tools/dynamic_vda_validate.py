"""Run the same universal VDA pair against the untouched FP32 reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import onnx
import export_guides as tools


def graph_evidence(directory):
    reports = []
    for label in ('init', 'step'):
        path = directory / f'vda_small_dynamic_{label}.onnx'
        model = onnx.load(path)
        producers = {o: n for n in model.graph.node for o in n.output}
        def ancestors(value, seen=None):
            seen = set() if seen is None else seen
            if value in seen:
                return set()
            seen.add(value)
            node = producers.get(value)
            if node is None:
                return {value}
            return {node.op_type} | set().union(*(ancestors(i, seen) for i in node.input))
        signatures = {}
        for value in list(model.graph.input) + list(model.graph.output):
            dims = [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]
            signatures[value.name] = dims
            if value.name in ('rgb', 'depth'):
                assert all(isinstance(dims[i], str) for i in (2, 3)), signatures
            else:
                assert isinstance(dims[0], str), signatures
        cubic = [n for n in model.graph.node if n.op_type == 'Resize' and
                 any(a.name == 'mode' and a.s == b'cubic' for a in n.attribute)]
        assert len(cubic) == 1
        dependencies = ancestors(cubic[0].input[2])
        assert {'rgb', 'Shape', 'Where'}.issubset(dependencies), dependencies
        spatial_ops = []
        for n in model.graph.node:
            if n.op_type in ('Resize', 'Reshape'):
                control = n.input[1:] if n.op_type == 'Resize' else n.input[1:2]
                deps = set().union(*(ancestors(i) for i in control))
                entry = {'node': n.name, 'op': n.op_type, 'shape_driven': 'Shape' in deps,
                         'rgb_dependent': 'rgb' in deps}
                if 'Shape' not in deps:
                    constants = []
                    for name in control:
                        node = producers.get(name)
                        if node is not None and node.op_type == 'Constant':
                            constants.extend(onnx.numpy_helper.to_array(a.t).tolist()
                                             for a in node.attribute if a.name == 'value')
                    allowed = [[-1], [1, -1, 384], [0, 32, -1]] if n.op_type == 'Reshape' else [[], [1., 1., 2., 2.]]
                    assert constants and all(c in allowed for c in constants), (n.name, constants)
                    entry['constant_controls'] = constants
                    entry['reason'] = 'inferred/copy axes or architectural group count' if n.op_type == 'Reshape' else 'relative 2x resize, not fixed spatial size'
                spatial_ops.append(entry)
        reports.append({'file': path.name, 'sha256': tools.sha(path), 'signatures': signatures,
                        'positional_resize_scale_ancestors': sorted(dependencies), 'spatial_operations': spatial_ops})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=tools.ROOT / 'model/native_guides')
    parser.add_argument('--native-exe', type=Path, default=tools.ROOT / 'native/guides/build/Release/guides_verify.exe')
    parser.add_argument('--graph-only', action='store_true')
    args = parser.parse_args()
    evidence = graph_evidence(args.output)
    (args.output / 'dynamic_vda_graph_evidence.json').write_text(json.dumps(evidence, indent=2))
    if args.graph_only:
        print('PASS graph spatial controls:', [(r['file'], len(r['spatial_operations'])) for r in evidence])
        return
    cases = [(512, 512, 518, 2), (371, 533, 518, 2), (533, 371, 518, 2),
             (237, 419, 518, 2), (99, 143, 98, 34), (101, 101, 98, 2)]
    reports = []
    for h, w, size, frames in cases:
        command = [sys.executable, str(Path(tools.__file__)), 'validate', '--component', 'depth',
                   '--height', str(h), '--width', str(w), '--depth-input-size', str(size),
                   '--frames', str(frames), '--output', str(args.output), '--native-exe', str(args.native_exe)]
        subprocess.run(command, check=True)
        report = args.output / f'validation_{h}x{w}_d{size}/comparison.json'
        reports.append(json.loads(report.read_text()))
        assert [tools.sha(args.output / r['file']) for r in evidence] == [r['sha256'] for r in evidence]
    (args.output / 'dynamic_vda_comparison.json').write_text(json.dumps({'graphs': evidence,
        'cases': reports, 'pass': all(r['pass'] for r in reports)}, indent=2))


if __name__ == '__main__':
    main()
