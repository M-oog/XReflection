from xreflection.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
from collections import OrderedDict
from xreflection.archs.dsrnet.lrm import LRM

from xreflection.archs.quantize_arch import QuantConv2d, build_conv


class QuantArgs:
    def __init__(self, **kwargs):
        # 设置默认值
        self.quantize = False
        self.quantize_a = 8.0
        self.quantize_w = 8.0
        self.quantizer = 'percentile'
        self.quantizer_w = 'omse'
        self.percentile_alpha = 0.999
        self.ema_beta = 0.99
        self.wbc = True
        # 用传入的字典覆盖默认值
        self.__dict__.update(kwargs)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels, args=None):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(channels, channels, 1)
            build_conv(channels, channels, 1, bias=True, args=args)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)


class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=False, args=None):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                # nn.Conv2d(c, c * 2, 1),
                # nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
                build_conv(c, c * 2, 1, args=args),
                build_conv(c * 2, c * 2, 3, padding=1, groups=c * 2, args=args)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c, args=args)),
            DualStreamBlock(build_conv(c, c, 1, args=args))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r


class FeaturePyramidVGG(nn.Module):
    def __init__(self, out_channels=64, shared_b=False, args=None):
        super().__init__()
        self.device = 'cuda'
        self.block5 = DualStreamSeq(
            MuGIBlock(512, shared_b, args=args),
            DualStreamBlock(nn.UpsamplingBilinear2d(scale_factor=2.0)),
        )

        self.block4 = DualStreamSeq(
            MuGIBlock(512, shared_b, args=args)
        )

        self.ch_map4 = DualStreamSeq(
            DualStreamBlock(
                # nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                build_conv(1024, 1024, 1, args=args),
                nn.PixelShuffle(2)),
            MuGIBlock(256, shared_b, args=args)
        )

        self.block3 = DualStreamSeq(
            MuGIBlock(256, shared_b, args=args)
        )

        self.ch_map3 = DualStreamSeq(
            DualStreamBlock(
                # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
                build_conv(512, 512, 1, args=args),
                nn.PixelShuffle(2)),
            MuGIBlock(128, shared_b, args=args)
        )

        self.block2 = DualStreamSeq(
            MuGIBlock(128, shared_b, args=args)
        )

        self.ch_map2 = DualStreamSeq(
            DualStreamBlock(
                # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                build_conv(256, 256, 1, args=args),
                nn.PixelShuffle(2)),
            MuGIBlock(64, shared_b, args=args)
        )

        self.block1 = DualStreamSeq(
            MuGIBlock(64, shared_b, args=args),
        )

        self.ch_map1 = DualStreamSeq(
            DualStreamBlock(build_conv(128, 128, 1, args=args)),
            MuGIBlock(128, shared_b, args=args),
            DualStreamBlock(build_conv(128, 32, 1, args=args)),
            MuGIBlock(32, shared_b, args=args),
        )

        self.block_intro = DualStreamSeq(
            DualStreamBlock(build_conv(3, 32, 3, padding=1, args=args)),
            MuGIBlock(32, shared_b, args=args)
        )

        self.ch_map0 = DualStreamBlock(
            # nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
            build_conv(64, out_channels, 1, args=args)
        )

    def forward(self, inp, vgg_feats):
        # 64,128,256,512,512
        vf1, vf2, vf3, vf4, vf5 = vgg_feats
        # 512=>512,512=>512
        f5_l, f5_r = self.block5(vf5)
        f4_l, f4_r = self.block4(vf4)
        f4_l, f4_r = self.ch_map4(torch.cat([f5_l, f4_l], dim=1), torch.cat([f5_r, f4_r], dim=1))
        # 256 => 256
        f3_l, f3_r = self.block3(vf3)
        # (256+256,256+256)->(128,128)
        f3_l, f3_r = self.ch_map3(torch.cat([f4_l, f3_l], dim=1), torch.cat([f4_r, f3_r], dim=1))
        # (128+128,128+128)->(64,64)
        f2_l, f2_r = self.block2(vf2)
        f2_l, f2_r = self.ch_map2(torch.cat([f3_l, f2_l], dim=1), torch.cat([f3_r, f2_r], dim=1))
        # (64+64,64+64)->(32,32)
        f1_l, f1_r = self.block1(vf1)
        f1_l, f1_r = self.ch_map1(torch.cat([f2_l, f1_l], dim=1), torch.cat([f2_r, f1_r], dim=1))
        # 64
        f0_l, f0_r = self.block_intro(inp, inp)
        f0_l, f0_r = self.ch_map0(torch.cat([f1_l, f0_l], dim=1), torch.cat([f1_r, f0_r], dim=1))
        return f0_l, f0_r


@ARCH_REGISTRY.register()
class DSRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=48, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[], lrm_blk_nums=[], shared_b=False,
                 quant_config=None):
        super().__init__()

        # ================== 统一转换逻辑 ==================
        if quant_config is None:
            # 情况1：没传参 -> 默认关闭量化
            self.quant_args = QuantArgs(quantize=False)

        elif isinstance(quant_config, dict):
            # 情况2：传入字典 (来自 YAML) -> 转为对象
            self.quant_args = QuantArgs(**quant_config)

        elif hasattr(quant_config, 'quantize'):
            # 情况3：传入对象 (来自 calibrate.py) -> 直接使用
            self.quant_args = quant_config

        else:
            raise ValueError(f"Unsupported type for quant_config: {type(quant_config)}")
        # =================================================

        self.intro = FeaturePyramidVGG(width, shared_b, args=self.quant_args)
        self.ending = DualStreamBlock(build_conv(width, out_channels, 3, padding=1, args=self.quant_args))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.lrm = LRM(width, num_blocks=lrm_blk_nums, args=self.quant_args)

        c = width
        for num in enc_blk_nums:
            self.encoders.append(
                DualStreamSeq(
                    *[MuGIBlock(c, shared_b, args=self.quant_args) for _ in range(num)]
                )
            )
            self.downs.append(
                DualStreamBlock(
                    # nn.Conv2d(c, c * 2, 2, 2)
                    build_conv(c, c * 2, 2, stride=2, args=self.quant_args)
                )
            )
            c *= 2

        self.middle_blks = DualStreamSeq(
            *[MuGIBlock(c, shared_b, args=self.quant_args) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                DualStreamBlock(
                    # nn.Conv2d(c, c * 2, 1, bias=False),
                    build_conv(c, c * 2, 1, bias=False, args=self.quant_args),
                    nn.PixelShuffle(2)
                )
            )
            c //= 2

            self.decoders.append(
                DualStreamSeq(
                    *[MuGIBlock(c, shared_b, args=self.quant_args) for _ in range(num)]
                )
            )

        # ================= 冻结权重逻辑 (参考 AdaBM) =================
        # 如果开启了量化，且不是在做校准 (通常校准时也不需要梯度，这里主要针对训练阶段)
        if self.quant_args.quantize:
            self._freeze_for_partial_qat()

    def _freeze_for_partial_qat(self):
        """
        参考 trainer-AdaBM.py 的逻辑：
        只允许量化参数 (lower_a, upper_a, upper_w) 更新，
        冻结所有卷积权重 (weight) 和偏置 (bias)。
        """
        print("=== [Config] Partial QAT Strategy Enabled ===")
        print("INFO: Freezing CNN weights. Only training quantization parameters (Step Sizes).")

        trainable_params = 0
        frozen_params = 0

        for name, param in self.named_parameters():
            # 判断是否为量化参数
            # 在 quantize_arch.py 中，我们定义了 lower_a, upper_a, upper_w
            is_quant_param = 'lower_a' in name or 'upper_a' in name or 'upper_w' in name

            # 此外，DSRNet 还有一些特定的标量参数 (a_l, a_r, b_l...) 用于双流融合
            # 建议也放开这些参数的训练，因为它们只占极小的显存，却对融合至关重要
            is_fusion_param = name.endswith('.a_l') or name.endswith('.a_r') or \
                              name.endswith('.b_l') or name.endswith('.b_r') or \
                              name.endswith('.a') or name.endswith('.b')

            if is_quant_param or is_fusion_param:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                # 冻结卷积权重、BN层参数等
                param.requires_grad = False
                frozen_params += param.numel()

        print(f"INFO: Frozen Params: {frozen_params / 1e6:.2f}M, Trainable Params: {trainable_params / 1e6:.2f}M")
        print("=============================================")

    def forward(self, inp, feats_inp):

        *_, H, W = inp.shape
        x, y = self.intro(inp, feats_inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x, y = encoder(x, y)
            encs.append((x, y))
            x, y = down(x, y)

        x, y = self.middle_blks(x, y)

        for decoder, up, (enc_x_skip, enc_y_skip) in zip(self.decoders, self.ups, encs[::-1]):
            x, y = up(x, y)
            x, y = x + enc_x_skip, y + enc_y_skip
            x, y = decoder(x, y)

        rr = self.lrm(x, y)

        x, y = self.ending(x, y)

        return x, y, rr
