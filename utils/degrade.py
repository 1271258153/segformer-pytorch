"""
可控退化工具，用于鲁棒性 / 退化分析实验。

在不重新训练的前提下，对输入图像或模型权重施加强度可控的扰动，
从而测出 mIoU 随退化强度变化的曲线。

退化分两类：
  1. 输入退化：模拟成像链路上的真实劣化（噪声、失焦、压缩、光照、分辨率）。
  2. 权重扰动：给权重加相对高斯噪声，用于考察解收敛点的平坦程度。
"""
import io

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

#--------------------------------------------------------------------#
#   每种退化的物理参数及其 severity 1~5 的取值
#   参考 ImageNet-C 的分级思路，数值按红外分割任务的尺度做了收敛
#--------------------------------------------------------------------#
CORRUPTION_PRESETS = {
    'gaussian_noise': {'param': 'sigma',   'levels': [5, 10, 20, 35, 50]},
    'gaussian_blur':  {'param': 'radius',  'levels': [0.5, 1.0, 2.0, 3.0, 4.0]},
    'jpeg':           {'param': 'quality', 'levels': [80, 60, 40, 25, 15]},
    'brightness':     {'param': 'factor',  'levels': [0.85, 0.70, 0.55, 0.40, 0.25]},
    'contrast':       {'param': 'factor',  'levels': [0.85, 0.70, 0.55, 0.40, 0.25]},
    'downscale':      {'param': 'factor',  'levels': [1.5, 2.0, 3.0, 4.0, 6.0]},
}

CORRUPTION_TYPES = list(CORRUPTION_PRESETS.keys())


def resolve_param(kind, severity=3, param=None):
    """给定退化类型，返回实际使用的物理参数值。显式的 param 优先于 severity。"""
    if kind not in CORRUPTION_PRESETS:
        raise ValueError("未知的退化类型 '%s'，可选：%s" % (kind, CORRUPTION_TYPES))
    if param is not None:
        return float(param)
    levels = CORRUPTION_PRESETS[kind]['levels']
    if not 1 <= int(severity) <= len(levels):
        raise ValueError("severity 需在 1~%d 之间，收到 %s" % (len(levels), severity))
    return float(levels[int(severity) - 1])


def param_name(kind):
    return CORRUPTION_PRESETS[kind]['param']


#--------------------------------------------------------------------#
#   各类输入退化，输入输出均为 PIL RGB 图像
#--------------------------------------------------------------------#
def _gaussian_noise(image, sigma, rng):
    arr = np.asarray(image, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _gaussian_blur(image, radius, rng):
    return image.filter(ImageFilter.GaussianBlur(radius))


def _jpeg(image, quality, rng):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _brightness(image, factor, rng):
    return ImageEnhance.Brightness(image).enhance(factor)


def _contrast(image, factor, rng):
    return ImageEnhance.Contrast(image).enhance(factor)


def _downscale(image, factor, rng):
    """先降到 1/factor 分辨率再插值回原尺寸，模拟有效分辨率的损失。"""
    w, h = image.size
    sw, sh = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
    small = image.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


_CORRUPTION_FNS = {
    'gaussian_noise': _gaussian_noise,
    'gaussian_blur':  _gaussian_blur,
    'jpeg':           _jpeg,
    'brightness':     _brightness,
    'contrast':       _contrast,
    'downscale':      _downscale,
}


def apply_corruption(image, kind, severity=3, param=None, rng=None):
    """
    对单张图施加一种输入退化。

    kind     : CORRUPTION_TYPES 之一，或 'none' 表示不做处理
    severity : 1~5 的等级，走 CORRUPTION_PRESETS 里的预设值
    param    : 直接指定物理参数，给定后覆盖 severity
    rng      : numpy 随机数发生器，传入可保证逐图可复现
    """
    if kind is None or kind == 'none':
        return image
    value = resolve_param(kind, severity, param)
    rng = rng if rng is not None else np.random.default_rng(0)
    return _CORRUPTION_FNS[kind](image.convert('RGB'), value, rng)


#--------------------------------------------------------------------#
#   权重扰动
#--------------------------------------------------------------------#
def perturb_weights(model, sigma, seed=11, only_conv_linear=True):
    """
    给权重叠加相对高斯噪声： w <- w + sigma * std(w) * N(0, 1)

    噪声按每个张量自身的标准差缩放，避免不同层量纲差异导致扰动强度失衡，
    与 loss landscape 分析中常用的 filter normalization 思路一致。
    返回被扰动的张量个数。
    """
    if sigma is None or sigma <= 0:
        return 0
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    perturbed = 0
    with torch.no_grad():
        for _, tensor in model.named_parameters():
            # 偏置与 BN 的一维参数量纲特殊，默认不扰动
            if only_conv_linear and tensor.dim() < 2:
                continue
            std = tensor.detach().float().std()
            if not torch.isfinite(std) or std.item() == 0.0:
                continue
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
            tensor.add_(noise.to(tensor.device, tensor.dtype) * (sigma * std))
            perturbed += 1
    return perturbed


#--------------------------------------------------------------------#
#   实验标识
#--------------------------------------------------------------------#
def build_tag(kind, severity, param, weight_noise, user_tag=None):
    """把退化设置压成一个可用作目录名的短标识。无退化时返回 'clean'。"""
    if user_tag:
        return str(user_tag)
    parts = []
    if kind and kind != 'none':
        value = resolve_param(kind, severity, param)
        parts.append('%s-%s%g' % (kind, param_name(kind)[0], value))
    if weight_noise and weight_noise > 0:
        parts.append('wnoise%g' % weight_noise)
    return '_'.join(parts) if parts else 'clean'
