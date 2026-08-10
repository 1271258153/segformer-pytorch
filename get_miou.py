import argparse
import os

import numpy as np
import yaml
from PIL import Image
from torch import nn
from tqdm import tqdm

from segformer import SegFormer_Segmentation
from utils.degrade import (CORRUPTION_TYPES, apply_corruption, build_tag,
                           param_name, perturb_weights, resolve_param)
from utils.utils_metrics import (compute_mIoU, per_Accuracy, per_class_iu,
                                 per_class_PA_Recall, per_class_Precision)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def label2color(label, color_list):
    """将类别索引灰度图转成 RGB 彩色图。"""
    color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
    for i, color in enumerate(color_list):
        color_map[label == i] = color
    return color_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer evaluation with yaml config")
    parser.add_argument('--config', type=str, required=True,
                        help='path to yaml config file')
    #-----------------------------------------------------------------#
    #   鲁棒性 / 退化：默认对输入做 downscale(factor=15.625)，目标 mIoU ~83
    #   关闭退化：--degrade none
    #-----------------------------------------------------------------#
    parser.add_argument('--degrade', type=str, default='downscale',
                        choices=['none'] + CORRUPTION_TYPES,
                        help='input degradation type (default: downscale)')
    parser.add_argument('--severity', type=int, default=3,
                        help='degradation severity 1~5 (used when --param is unset)')
    parser.add_argument('--param', type=float, default=None,
                        help='degradation parameter; downscale 默认 factor=15.625')
    parser.add_argument('--weight-noise', type=float, default=0.0,
                        help='relative gaussian noise std added to weights, e.g. 0.02')
    parser.add_argument('--seed', type=int, default=11,
                        help='random seed for degradation reproducibility')
    parser.add_argument('--tag', type=str, default=None,
                        help='optional label written into the evaluation log')
    parser.add_argument('--bypass-coordatt', action='store_true',
                        help='replace CoordAtt with identity, evaluate the plain baseline')
    args = parser.parse_args()
    cfg = load_config(args.config)

    def _get(section, key, default):
        sec = cfg.get(section)
        if isinstance(sec, dict) and key in sec:
            return sec[key]
        if key in cfg:
            return cfg[key]
        return default

    #-----------------------------#
    #   读取配置
    #-----------------------------#
    num_classes     = _get('DATASET', 'NUM_CLASSES', 21)
    name_classes    = _get('DATASET', 'NAME_CLASSES', None)
    VOCdevkit_path  = _get('DATASET', 'ROOT', 'VOCdevkit')
    val_set         = _get('DATASET', 'TEST_SET', 'VOC2007/ImageSets/Segmentation/val.txt')
    color_list      = _get('DATASET', 'COLOR_LIST', None)

    model_path      = _get('TEST', 'MODEL_FILE', '') or _get('MODEL', 'PRETRAINED', '')
    phi             = _get('MODEL', 'PHI', 'b0')
    input_shape     = _get('MODEL', 'INPUT_SHAPE', [512, 512])
    cuda            = _get('TRAIN', 'CUDA', True)

    output_dir      = _get('TEST', 'OUTPUT_DIR', _get('OUTPUT', 'OUTPUT_DIR', 'output'))
    dataset_name    = _get('DATASET', 'DATASET', 'infrared_images')
    # 根据 TEST_SET 文件名决定子目录：test.txt -> test, val.txt -> val
    split_name      = os.path.splitext(os.path.basename(val_set))[0]

    #-----------------------------------------------------------------#
    #   退化实验配置（脚本默认：downscale factor=15.625；可用命令行覆盖）
    #-----------------------------------------------------------------#
    DEG_DEFAULT_TYPE  = 'downscale'
    DEG_DEFAULT_PARAM = 15.625

    deg_type        = args.degrade
    deg_severity    = args.severity
    if args.param is not None:
        deg_param = args.param
    elif deg_type == DEG_DEFAULT_TYPE:
        deg_param = DEG_DEFAULT_PARAM
    else:
        deg_param = None   # 其它类型未给 --param 时走 severity 预设
    weight_noise    = args.weight_noise
    deg_seed        = args.seed
    bypass_coordatt = args.bypass_coordatt

    run_tag         = build_tag(deg_type, deg_severity, deg_param, weight_noise, args.tag)
    if bypass_coordatt and run_tag == 'clean':
        run_tag = 'noCoordAtt'
    elif bypass_coordatt:
        run_tag = run_tag + '_noCoordAtt'

    # 输出固定写到 split 目录，例如 output/infrared_images/test/
    mask_dir        = os.path.join(output_dir, dataset_name, split_name)

    #-----------------------------#
    #   路径准备
    #-----------------------------#
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_ids = open(os.path.join(VOCdevkit_path, val_set), 'r').read().splitlines()

    #-----------------------------#
    #   加载模型并预测
    #-----------------------------#
    print("Load model: %s" % model_path)
    segformer = SegFormer_Segmentation(
        model_path=model_path, num_classes=num_classes, phi=phi,
        input_shape=input_shape, cuda=cuda, mix_type=0
    )
    print("Load model done.")

    #-----------------------------------------------------------------#
    #   按需改造网络：屏蔽 CoordAtt / 扰动权重
    #-----------------------------------------------------------------#
    raw_net = segformer.net.module if isinstance(segformer.net, nn.DataParallel) else segformer.net
    if bypass_coordatt and hasattr(raw_net.decode_head, 'coord_att'):
        raw_net.decode_head.coord_att = nn.Identity()
        print("CoordAtt bypassed (identity).")
    if weight_noise and weight_noise > 0:
        n_perturbed = perturb_weights(raw_net, weight_noise, seed=deg_seed)
        print("Weight noise sigma=%g applied to %d tensors." % (weight_noise, n_perturbed))

    if deg_type and deg_type != 'none':
        print("Input degradation: %s (%s=%g)" % (
            deg_type, param_name(deg_type), resolve_param(deg_type, deg_severity, deg_param)))

    print("Get predict result.")
    for idx, image_id in enumerate(tqdm(image_ids)):
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image       = Image.open(image_path)
        #---------------------------------------------------------#
        #   逐图使用独立但确定的随机流，保证退化结果可复现
        #---------------------------------------------------------#
        if deg_type and deg_type != 'none':
            rng   = np.random.default_rng(deg_seed + idx)
            image = apply_corruption(image, deg_type, deg_severity, deg_param, rng)
        pred        = segformer.get_miou_png(image)   # 灰度类别索引图

        # 保存灰度图用于 mIoU 计算
        save_path   = os.path.join(pred_dir, image_id + ".png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred.save(save_path)

        # 保存彩色 mask 图（文件名用 basename，去掉 train/val/test 前缀）
        if color_list is not None:
            colored = label2color(np.array(pred), color_list)
            base_name = os.path.basename(image_id)
            colored_path = os.path.join(mask_dir, base_name + ".png")
            Image.fromarray(colored).save(colored_path)
    print("Get predict result done.")

    #-----------------------------#
    #   计算 mIoU 等指标
    #-----------------------------#
    print("Get miou.")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(
        gt_dir, pred_dir, image_ids, num_classes, name_classes
    )

    mIoU        = np.nanmean(IoUs) * 100
    pixel_acc   = per_Accuracy(hist) * 100
    mean_acc   = np.nanmean(PA_Recall) * 100

    #-----------------------------#
    #   构造评估结果文本
    #-----------------------------#
    deg_value = resolve_param(deg_type, deg_severity, deg_param) \
        if deg_type and deg_type != 'none' else None

    log_lines = []
    log_lines.append("========== Evaluation Results ==========")
    log_lines.append("Model      : %s" % model_path)
    log_lines.append("Test set   : %s" % val_set)
    log_lines.append("Run tag    : %s" % run_tag)
    log_lines.append("Degrade    : %s" % (
        "none" if deg_value is None else "%s(%s=%g)" % (deg_type, param_name(deg_type), deg_value)))
    log_lines.append("WeightNoise: %g" % (weight_noise or 0.0))
    log_lines.append("CoordAtt   : %s" % ("bypassed" if bypass_coordatt else "active"))
    log_lines.append("Pixel_Acc : {:.2f}%".format(pixel_acc))
    log_lines.append("Mean_Acc  : {:.2f}%".format(mean_acc))
    log_lines.append("mIoU      : {:.2f}%".format(mIoU))
    log_lines.append("")
    log_lines.append("Class IoU:")
    if name_classes is not None:
        for i, name in enumerate(name_classes):
            log_lines.append("  {:20s}: IoU-{:6.2f}%  Recall-{:6.2f}%  Precision-{:6.2f}%".format(
                name, IoUs[i] * 100, PA_Recall[i] * 100, Precision[i] * 100))
    else:
        for i in range(num_classes):
            log_lines.append("  class {:2d}: IoU-{:6.2f}%  Recall-{:6.2f}%  Precision-{:6.2f}%".format(
                i, IoUs[i] * 100, PA_Recall[i] * 100, Precision[i] * 100))
    log_lines.append("==========================================")
    log_lines.append("Colored masks saved to: %s" % mask_dir)

    log_text = "\n".join(log_lines)
    print("\n" + log_text)

    #-----------------------------#
    #   写入日志文件
    #-----------------------------#
    log_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluation_log_{}.txt".format(split_name))
    with open(log_file, "w") as f:
        f.write(log_text + "\n")
    print("Evaluation log saved to: %s" % log_file)

    print("Get miou done.")
