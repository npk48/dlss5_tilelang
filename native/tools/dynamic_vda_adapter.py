"""Export-only adapters; the original reference modules/files remain untouched."""
import inspect
import textwrap
import types
import torch
from torch import nn


class PosResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embedding, scales):
        return torch.nn.functional.interpolate(embedding, scale_factor=tuple(scales[-2:].tolist()),
                                               mode='bicubic', align_corners=False, antialias=False)

    @staticmethod
    def symbolic(g, embedding, scales):
        return g.op('Resize', embedding, g.op('Constant', value_t=torch.tensor([], dtype=torch.float32)),
                    scales, mode_s='cubic', coordinate_transformation_mode_s='half_pixel',
                    cubic_coeff_a_f=-0.75, nearest_mode_s='floor')


def position(self, x, w, h):
    # Shape scalars stay in the graph, including the exact 37x37 identity case.
    dims = torch.stack(tuple(v if torch.is_tensor(v) else torch.tensor(v) for v in
                             (w // self.patch_size, h // self.patch_size))).to(torch.float32)
    base = torch.all(dims == 37)
    spatial = torch.where(base, torch.ones_like(dims), (dims + self.interpolate_offset) / 37.)
    scales = torch.cat((torch.ones(2, device=dims.device), spatial))
    patch = self.pos_embed[:, 1:].float().reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
    patch = PosResize.apply(patch, scales).permute(0, 2, 3, 1).reshape(1, -1, 384)
    return torch.cat((self.pos_embed[:, :1], patch), dim=1).to(x.dtype)


def rearrange_dynamic(x, pattern, **axes):
    if pattern == 'b c f h w -> (b f) c h w':
        return x.squeeze(2)
    if pattern == '(b f) c h w -> b c f h w':
        return x.unsqueeze(2)
    if pattern == '(b f) d c -> (b d) f c':
        return x.reshape(-1, axes['f'], x.shape[1], x.shape[2]).permute(0, 2, 1, 3).reshape(-1, axes['f'], x.shape[2])
    if pattern == '(b d) f c -> (b f) d c':
        return x.reshape(-1, axes['d'], x.shape[1], x.shape[2]).permute(0, 2, 1, 3).reshape(-1, axes['d'], x.shape[2])
    raise ValueError(pattern)


def adapt(network):
    from whitebox_pipeline._vda.dinov2 import DinoVisionTransformer
    from whitebox_pipeline._vda.dpt_temporal import DPTHeadTemporal
    for module in network.modules():
        forward = type(module).forward
        if 'rearrange' in forward.__code__.co_names:
            namespace = dict(forward.__globals__, rearrange=rearrange_dynamic)
            module.forward = types.MethodType(types.FunctionType(forward.__code__, namespace,
                forward.__name__, forward.__defaults__, forward.__closure__), module)
        if isinstance(module, DinoVisionTransformer):
            if module.interpolate_antialias or module.pos_embed.shape != (1, 1370, 384):
                raise ValueError('Adapter requires FP32 Small DINO 37x37 non-antialiased embeddings')
            module.interpolate_pos_encoding = types.MethodType(position, module)
        if isinstance(module, DPTHeadTemporal):
            # Reuse the exact installed reference forward, changing only the two
            # Python integer conversions which otherwise freeze output geometry.
            source = textwrap.dedent(inspect.getsource(DPTHeadTemporal.forward))
            assert source.count('int(patch_h * 14)') == 2
            source = source.replace('int(patch_h * 14)', 'patch_h * 14').replace('int(patch_w * 14)', 'patch_w * 14')
            # This export API is batch=frame_length=1. Avoid legacy ONNX
            # unflatten shape inference freezing subsequent aten::size values.
            source = source.replace('.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)', '.unsqueeze(2)')
            source = source.replace('.permute(0, 2, 1, 3, 4).flatten(0, 1)', '.squeeze(2)')
            namespace = dict(DPTHeadTemporal.forward.__globals__)
            exec(compile(source, __file__, 'exec'), namespace)
            module.forward = types.MethodType(namespace['forward'], module)
    return network
