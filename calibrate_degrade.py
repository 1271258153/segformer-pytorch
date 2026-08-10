"""
退化强度标定：为鲁棒性实验寻找落在目标 mIoU 区间内的退化参数。

模型只加载一次，通过二分搜索在指定退化类型的参数轴上定位工作点，
比反复调用 get_miou.py 快一个量级。

用法示例：
    python calibrate_degrade.py --config experiments/infrared_images/test.yaml \
        --degrade gaussian_noise --target-low 80 --target-high 84
    python calibrate_degrade.py --config experiments/infrared_images/test.yaml \
        --degrade gaussian_blur --scan 1,2,3,4,6,8

标定出参数后，用 get_miou.py 复现完整结果并留档：
    python get_miou.py --config ... --degrade gaussian_noise --param <值>
"""
import argparse
import os

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from segformer import SegFormer_Segmentation
from utils.degrade import CORRUPTION_TYPES, apply_corruption, param_name
from utils.utils_metrics import fast_hist, per_class_iu

#--------------------------------------------------------------------#
#   二分搜索的参数区间：下界基本无损，上界足以让模型明显退化
#   除亮度/对比度外，参数越大退化越强
#--------------------------------------------------------------------#
SEARCH_RANGE = {
    'gaussian_noise': (0.0, 200.0),
    'gaussian_blur':  (0.0, 20.0),
    'jpeg':           (100.0, 1.0),     # quality 越小越差，故区间反向
    'brightness':     (1.0, 0.02),
    'contrast':       (1.0, 0.02),
    'downscale':      (1.0, 40.0),
}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(segformer, records, num_classes, kind, param, seed):
    """在给定退化参数下跑完整测试集，返回 mIoU（百分数）。"""
    hist = np.zeros((num_classes, num_classes))
    for idx, (image_path, gt_path) in enumerate(records):
        image = Image.open(image_path)
        if kind != 'none':
            image = apply_corruption(image, kind, param=param,
                                     rng=np.random.default_rng(seed + idx))
        pred = np.array(segformer.get_miou_png(image))
        gt   = np.array(Image.open(gt_path))
        if pred.shape != gt.shape:
            continue
        hist += fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return float(np.nanmean(per_class_iu(hist)) * 100)


def main():
    parser = argparse.ArgumentParser(description="Calibrate degradation strength to a target mIoU")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--degrade', type=str, required=True, choices=CORRUPTION_TYPES)
    parser.add_argument('--target-low', type=float, default=80.0)
    parser.add_argument('--target-high', type=float, default=84.0)
    parser.add_argument('--scan', type=str, default=None,
                        help='comma separated parameter values, scan instead of bisect')
    parser.add_argument('--max-iter', type=int, default=12)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--bypass-coordatt', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)

    def _get(section, key, default):
        sec = cfg.get(section)
        if isinstance(sec, dict) and key in sec:
            return sec[key]
        return cfg.get(key, default)

    num_classes    = _get('DATASET', 'NUM_CLASSES', 21)
    VOCdevkit_path = _get('DATASET', 'ROOT', 'VOCdevkit')
    val_set        = _get('DATASET', 'TEST_SET', 'VOC2007/ImageSets/Segmentation/val.txt')
    model_path     = _get('TEST', 'MODEL_FILE', '') or _get('MODEL', 'PRETRAINED', '')
    phi            = _get('MODEL', 'PHI', 'b0')
    input_shape    = _get('MODEL', 'INPUT_SHAPE', [512, 512])
    cuda           = _get('TRAIN', 'CUDA', True)

    image_ids = open(os.path.join(VOCdevkit_path, val_set), 'r').read().splitlines()
    records   = [(os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + i + ".jpg"),
                  os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/" + i + ".png"))
                 for i in image_ids]

    segformer = SegFormer_Segmentation(
        model_path=model_path, num_classes=num_classes, phi=phi,
        input_shape=input_shape, cuda=cuda, mix_type=0
    )
    if args.bypass_coordatt:
        from torch import nn
        raw = segformer.net.module if isinstance(segformer.net, nn.DataParallel) else segformer.net
        if hasattr(raw.decode_head, 'coord_att'):
            raw.decode_head.coord_att = nn.Identity()
            print("CoordAtt bypassed.")

    kind  = args.degrade
    pname = param_name(kind)

    clean = evaluate(segformer, records, num_classes, 'none', None, args.seed)
    print("\n[clean] mIoU = %.2f%%  (%d images)\n" % (clean, len(records)))

    #----------------------------------------------------------------#
    #   扫描模式：给定若干参数值，直接输出退化曲线
    #----------------------------------------------------------------#
    if args.scan:
        values = [float(v) for v in args.scan.split(',')]
        print("%-12s %10s" % (pname, 'mIoU(%)'))
        print("-" * 24)
        for value in values:
            miou = evaluate(segformer, records, num_classes, kind, value, args.seed)
            print("%-12g %10.2f" % (value, miou))
        return

    #----------------------------------------------------------------#
    #   二分模式：在参数轴上定位落入目标区间的工作点
    #----------------------------------------------------------------#
    low, high = SEARCH_RANGE[kind]
    target    = 0.5 * (args.target_low + args.target_high)
    best      = None
    history   = []

    print("Bisecting %s on %s to reach mIoU %.1f~%.1f ...\n"
          % (kind, pname, args.target_low, args.target_high))
    for step in range(args.max_iter):
        mid  = 0.5 * (low + high)
        miou = evaluate(segformer, records, num_classes, kind, mid, args.seed)
        history.append((mid, miou))
        hit  = args.target_low <= miou <= args.target_high
        print("  step %2d  %s=%-10.4g mIoU=%6.2f%%%s" % (step + 1, pname, mid, miou, '  <-- 命中' if hit else ''))
        if hit and (best is None or abs(miou - target) < abs(best[1] - target)):
            best = (mid, miou)
        if best is not None and abs(miou - target) < 0.3:
            break
        # low 端对应弱退化(高 mIoU)，high 端对应强退化(低 mIoU)
        if miou > args.target_high:
            low = mid
        else:
            high = mid

    print()
    if best is None:
        print("未能在 %d 次迭代内落入目标区间，最接近的结果：" % args.max_iter)
        closest = min(history, key=lambda kv: abs(kv[1] - target))
        print("  %s=%g -> mIoU=%.2f%%" % (pname, closest[0], closest[1]))
        print("可尝试换一种退化类型，或用 --scan 手动探查参数范围。")
    else:
        print("标定结果：%s 取 %s=%g 时 mIoU=%.2f%%（clean 为 %.2f%%，下降 %.2f 分）"
              % (kind, pname, best[0], best[1], clean, clean - best[1]))
        print("\n复现完整评估并留档：")
        print("  python get_miou.py --config %s --degrade %s --param %g"
              % (args.config, kind, best[0]))


if __name__ == "__main__":
    main()
