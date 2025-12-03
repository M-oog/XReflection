import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        return torch.round(x_in)

    @staticmethod
    def backward(ctx, g):
        return g, None


class QuantConv2d(nn.Conv2d):
    """
    针对 DSRNet 优化的量化卷积层。
    保留了 quantize.py 中的核心算法 (OMSE, Percentile, EMA)，但接口适配为标准 nn.Module。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, args=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.args = args

        # 1. 获取量化位宽配置 (默认 8bit)
        self.a_bit = getattr(args, 'quantize_a', 8.0)
        self.w_bit = getattr(args, 'quantize_w', 8.0)

        # 2. 定义量化参数
        # 激活值量化参数 (Per-Tensor): shape [1, C, 1, 1] 用于广播，但通常激活是 Per-Layer 统计，这里简化为 Per-Tensor scalar 或 Per-Channel
        # 原 quantize.py 中 lower_a 是 [1, In, 1, 1]，这里为了通用性，我们先初始化为 [1, 1, 1, 1] (Per-layer) 或 [1, In, 1, 1]
        # 通常激活量化使用 Per-Layer (scalar) 较多，但为了兼容原代码逻辑，我们保留 shape 灵活性
        self.lower_a = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.upper_a = nn.Parameter(torch.ones(1, 1, 1, 1))

        # 权重量化参数 (Per-Channel): shape [Out, 1, 1, 1]
        self.upper_w = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        self.round_func = Round.apply

        # 3. 状态控制
        self.init_mode = False  # 是否处于初始化/校准模式
        self.ema_epoch = 0  # 用于 EMA 初始化的计数器

    def init_qparams_a(self, x):
        """初始化激活值量化参数 (Percentile / MinMax)"""
        # x shape: [B, C, H, W]
        if self.args and getattr(self.args, 'quantizer', '') == 'minmax':
            # MinMax: 全局最小最大
            cur_min = x.detach().min()
            cur_max = x.detach().max()
            lower_a = cur_min.view(1, 1, 1, 1)
            upper_a = cur_max.view(1, 1, 1, 1)

        elif self.args and getattr(self.args, 'quantizer', '') == 'percentile':
            # Percentile: 去除离群点
            alpha = getattr(self.args, 'percentile_alpha', 0.9999)

            x_detach = x.detach()
            # ================= [修改开始] =================
            # 解决 RuntimeError: quantile() input tensor is too large
            # 策略：如果元素数量超过 500万，就进行降采样
            # 统计学上，100万个采样点足够拟合 99.9% 的分布
            MAX_ELEMENTS = 5_000_000
            if x_detach.numel() > MAX_ELEMENTS:
                # 计算步长，保持采样后依然有足够的数据点
                # 假设 B, C 维度保留，只在 H, W 上降采样
                step = int((x_detach.numel() / MAX_ELEMENTS) ** 0.5)
                step = max(1, step)
                # 使用切片采样 [:, :, ::step, ::step]
                x_sampled = x_detach[:, :, ::step, ::step].reshape(-1)
            else:
                x_sampled = x_detach.view(-1)
            # ================= [修改结束] =================

            lower_a = torch.quantile(x_sampled, 1 - alpha)
            upper_a = torch.quantile(x_sampled, alpha)
            lower_a = lower_a.view(1, 1, 1, 1)
            upper_a = upper_a.view(1, 1, 1, 1)

        else:
            # Default MinMax
            lower_a = x.detach().min().view(1, 1, 1, 1)
            upper_a = x.detach().max().view(1, 1, 1, 1)

        # EMA 更新逻辑
        if self.ema_epoch == 0:
            self.lower_a.data.copy_(lower_a)
            self.upper_a.data.copy_(upper_a)
        else:
            beta = getattr(self.args, 'ema_beta', 0.99)
            self.lower_a.data.copy_(self.lower_a.data * beta + lower_a * (1 - beta))
            self.upper_a.data.copy_(self.upper_a.data * beta + upper_a * (1 - beta))

        self.ema_epoch += 1

    def init_qparams_w(self):
        """初始化权重量化参数 (OMSE / MinMax)"""
        # weight shape: [Out, In, k, k]
        w = self.weight
        w_flat = w.view(w.size(0), -1)  # [Out, N]

        quantizer_w = getattr(self.args, 'quantizer_w', 'minmax')

        if quantizer_w == 'omse':
            # === 高效并行 Channel-wise OMSE (移植自 quantize.py) ===
            max_val = w_flat.abs().max(dim=1)[0]  # [Out]
            best_score = torch.full_like(max_val, float('inf'))
            best_upper = max_val.clone()

            # 搜索 100 个截断阈值
            for i in range(100):
                ratio = 1.0 - i * 0.01
                curr_upper = max_val * ratio
                curr_upper_view = curr_upper.view(-1, 1)  # [Out, 1]

                step_size = curr_upper_view / (2 ** self.w_bit - 1 + 1e-6)

                # 模拟截断和量化
                w_c = torch.clamp(w_flat, min=-curr_upper_view, max=curr_upper_view)
                w_q = torch.round(w_c / step_size) * step_size

                # 计算 MSE
                score = (w_flat - w_q).abs().pow(2).mean(dim=1)  # [Out]

                # 更新最佳值
                mask = score < best_score
                best_score = torch.where(mask, score, best_score)
                best_upper = torch.where(mask, curr_upper, best_upper)

            upper_w = best_upper

        else:
            # MinMax Fallback
            upper_w = w_flat.abs().max(dim=1)[0]

        # 赋值
        self.upper_w.data.copy_(upper_w.view(-1, 1, 1, 1))

    def forward(self, x):
        # 1. 初始化逻辑
        # 只有在 self.init_mode = True 时才执行统计
        if self.init_mode:
            self.init_qparams_a(x)
            # 权重只需要初始化一次 (在第1个 epoch 的第1个 batch)
            if self.ema_epoch == 1:
                self.init_qparams_w()

        # 2. 激活值量化
        # Clamp -> Normalize -> Quantize -> De-normalize
        x_c = torch.clamp(x, min=self.lower_a, max=self.upper_a)
        range_a = self.upper_a - self.lower_a
        x_norm = (x_c - self.lower_a) / (range_a + 1e-6)

        scale_a = 2 ** self.a_bit - 1
        x_q_norm = self.round_func(x_norm * scale_a) / scale_a

        x_q = x_q_norm * range_a + self.lower_a

        # 3. 权重量化
        # Symmetric quantization: [-upper, upper]
        w_c = torch.clamp(self.weight, min=-self.upper_w, max=self.upper_w)
        scale_w = (2 * self.upper_w) / (2 ** self.w_bit - 1 + 1e-6)

        w_int = self.round_func(w_c / scale_w)
        w_q = w_int * scale_w

        # [补全] 权重偏差校正 (Weight Bias Correction)
        # 仅在参数 wbc=True 时生效
        if getattr(self.args, 'wbc', False):
            # 计算全精度权重的均值和方差
            mean_fp = self.weight.mean(dim=(1, 2, 3), keepdim=True)
            var_fp = self.weight.var(dim=(1, 2, 3), keepdim=True)

            # 计算量化权重的均值和方差
            mean_q = w_q.mean(dim=(1, 2, 3), keepdim=True)
            var_q = w_q.var(dim=(1, 2, 3), keepdim=True)

            # 修正公式: w'_q = (sigma_fp / sigma_q) * (w_q - mu_q) + mu_fp
            # 加上 eps 防止除零
            eps = 1e-6
            scale = torch.sqrt((var_fp + eps) / (var_q + eps))
            w_q = scale * (w_q - mean_q) + mean_fp

        # 4. 执行卷积
        return F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def build_conv(in_c, out_c, k_size, stride=1, padding=0, groups=1, bias=True, args=None):
    """
    根据 args.quantize 决定构建普通卷积还是量化卷积
    """
    if args is not None and getattr(args, 'quantize', False):
        return QuantConv2d(in_c, out_c, k_size, stride=stride, padding=padding,
                           groups=groups, bias=bias, args=args)
    else:
        return nn.Conv2d(in_c, out_c, k_size, stride=stride, padding=padding,
                         groups=groups, bias=bias)
