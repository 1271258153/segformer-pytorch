# -*- coding: utf-8 -*-
"""生成四宫格对比图（原图 | 真值 | 预测 | 叠加）或仅生成叠加图。

用法：
    python -B tools/make_comparison.py \
        --config experiments/infrared_images/test.yaml

    # 只为指定图片生成叠加图（可传图片 ID 或带扩展名的文件名）
    python -B tools/make_comparison.py \
        --config experiments/infrared_images/test.yaml \
        --input-dir VOCdevkit/VOC2007/JPEGImages \
        --pred-dir output/infrared_images/test \
        --output-dir output/overlays \
        --overlay-only --color-weight 0.7 --background-depth 0.3 \
        --images image_001 image_002.jpg

默认读取 get_miou.py 生成的彩色预测图，并将对比图保存到
output/infrared_images/comparison_images/。
"""
import argparse
import os

import cv2
import numpy as np
import yaml
from PIL import Image


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config_value(config, section, key, default=None):
    section_config = config.get(section)
    if isinstance(section_config, dict) and key in section_config:
        return section_config[key]
    return config.get(key, default)


def label2color(label, color_list):
    color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
    for class_id, color in enumerate(color_list):
        color_map[label == class_id] = color
    return color_map


def overlay(image, color_mask, alpha=0.5):
    return (image * (1.0 - alpha) + color_mask * alpha).astype(np.uint8)


def overlay_foreground(
    image,
    color_mask,
    alpha=0.5,
    background_color=(0, 0, 0),
    background_depth=0.0,
):
    """混合预测前景，并按需压暗背景。"""
    result = (image * (1.0 - background_depth)).astype(np.uint8)
    foreground = np.any(
        color_mask != np.asarray(background_color, dtype=np.uint8), axis=2
    )
    result[foreground] = overlay(
        image[foreground], color_mask[foreground], alpha=alpha
    )
    return result


def hconcat(images, gap=4, gap_color=(255, 255, 255)):
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), gap_color, dtype=np.uint8)

    x = 0
    for image in images:
        canvas[:image.shape[0], x:x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


def normalize_image_id(image_name):
    """将图片 ID 或文件名转换为数据集中的图片 ID。"""
    image_id, extension = os.path.splitext(image_name)
    if extension.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        return image_id
    return image_name


def main():
    parser = argparse.ArgumentParser(description="Generate comparison or overlay images")
    parser.add_argument(
        "--config",
        default="experiments/infrared_images/test.yaml",
        help="测试 yaml 配置文件",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="彩色预测 mask 目录；默认由配置中的输出目录、数据集名和测试集推导",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="原图输入目录；默认使用数据集的 VOC2007/JPEGImages 目录",
    )
    parser.add_argument(
        "--out-dir", "--output-dir",
        default="output/infrared_images/comparison_images",
        help="生成图片的输出目录",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="只生成原图与预测 mask 的叠加图，不生成四宫格",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        metavar="IMAGE",
        help="只处理指定图片；可传一个或多个图片 ID/文件名",
    )
    parser.add_argument("--alpha", default=0.5, type=float, help="叠加图中预测 mask 的权重")
    parser.add_argument(
        "--color-weight", "--overlay-color-weight",
        dest="color_weight",
        default=None,
        type=float,
        help=(
            "仅在 --overlay-only 模式下使用的前景颜色权重（0 到 1）；"
            "默认使用 --alpha 的值"
        ),
    )
    parser.add_argument(
        "--background-depth", "--background-darkness",
        dest="background_depth",
        default=0.0,
        type=float,
        help=(
            "仅在 --overlay-only 模式下使用的背景压暗程度（0 到 1）；"
            "0 保持原图，1 为全黑"
        ),
    )
    parser.add_argument("--gap", default=4, type=int, help="各子图之间的白色间隔（像素）")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha 必须在 0 到 1 之间")
    if args.color_weight is not None and not 0.0 <= args.color_weight <= 1.0:
        parser.error("--color-weight 必须在 0 到 1 之间")
    if args.color_weight is not None and not args.overlay_only:
        parser.error("--color-weight 只能与 --overlay-only 一起使用")
    if not 0.0 <= args.background_depth <= 1.0:
        parser.error("--background-depth 必须在 0 到 1 之间")
    if args.background_depth != 0.0 and not args.overlay_only:
        parser.error("--background-depth 只能与 --overlay-only 一起使用")
    if args.gap < 0:
        parser.error("--gap 不能小于 0")

    config = load_config(args.config)
    dataset_root = get_config_value(config, "DATASET", "ROOT", "VOCdevkit")
    test_set = get_config_value(
        config, "DATASET", "TEST_SET", "VOC2007/ImageSets/Segmentation/test.txt"
    )
    dataset_name = get_config_value(config, "DATASET", "DATASET", "infrared_images")
    color_list = get_config_value(config, "DATASET", "COLOR_LIST")
    output_dir = get_config_value(config, "TEST", "OUTPUT_DIR", "output")

    if not color_list and not args.overlay_only:
        raise ValueError("配置文件 DATASET.COLOR_LIST 不能为空")

    split_name = os.path.splitext(os.path.basename(test_set))[0]
    pred_dir = args.pred_dir or os.path.join(output_dir, dataset_name, split_name)
    list_path = os.path.join(dataset_root, test_set)
    image_dir = args.input_dir or os.path.join(dataset_root, "VOC2007", "JPEGImages")
    label_dir = os.path.join(dataset_root, "VOC2007", "SegmentationClass")

    if args.images:
        image_ids = [normalize_image_id(image_name) for image_name in args.images]
    else:
        with open(list_path, "r") as f:
            image_ids = [line.strip().split()[0] for line in f if line.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    saved = 0
    missing = 0
    for image_id in image_ids:
        base_name = os.path.basename(image_id)
        image_path = os.path.join(image_dir, image_id + ".jpg")
        pred_path = os.path.join(pred_dir, base_name + ".png")
        required_paths = [image_path, pred_path]
        if not args.overlay_only:
            label_path = os.path.join(label_dir, image_id + ".png")
            required_paths.append(label_path)

        if not all(os.path.isfile(path) for path in required_paths):
            print("skip missing input: {}".format(image_id))
            missing += 1
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        pred = cv2.imread(pred_path, cv2.IMREAD_COLOR)

        height, width = image.shape[:2]
        if pred.shape[:2] != (height, width):
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_color = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        if args.overlay_only:
            color_weight = (
                args.alpha if args.color_weight is None else args.color_weight
            )
            background_color = color_list[0] if color_list else (0, 0, 0)
            blended = overlay_foreground(
                image_rgb,
                pred_color,
                alpha=color_weight,
                background_color=background_color,
                background_depth=args.background_depth,
            )
        else:
            blended = overlay(image_rgb, pred_color, alpha=args.alpha)

        result = blended
        if not args.overlay_only:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label.shape[:2] != (height, width):
                label = cv2.resize(
                    label, (width, height), interpolation=cv2.INTER_NEAREST
                )
            gt_color = label2color(label, color_list)
            result = hconcat(
                [image_rgb, gt_color, pred_color, blended], gap=args.gap
            )

        Image.fromarray(result).save(os.path.join(args.out_dir, base_name + ".png"))
        saved += 1

    print("done: {} saved to {}, {} missing inputs".format(saved, args.out_dir, missing))


if __name__ == "__main__":
    main()
