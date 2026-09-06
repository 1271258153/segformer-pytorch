#!/usr/bin/env python3
"""计算 SegFormer 参数量和端到端推理 FPS。

使用方法：
    # 默认使用 SegFormer-B0、10 类、640x640、batch size=1
    python tools/get_fps_and_pm.py --phi b0 --device cuda

    # 使用训练后的权重测试
    python tools/get_fps_and_pm.py --checkpoint logs/best_epoch_weights.pth

    # 指定图片目录和输入尺寸
    python tools/get_fps_and_pm.py \
        --image-dir datasets/infrared_images/images/test \
        --height 640 --width 640

    # 使用 CUDA FP16 测试
    python tools/get_fps_and_pm.py --device cuda --fp16

计时范围：磁盘读图与解码、缩放、归一化、CPU 到 GPU 传输、模型 forward、
argmax，以及最终分割图传回 CPU。不包含模型构建、权重加载和结果保存。
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nets.segformer import SegFormer  # noqa: E402


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile SegFormer parameters and end-to-end inference FPS')
    parser.add_argument('--phi', default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer backbone variant')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='number of segmentation classes')
    parser.add_argument('--height', default=640, type=int,
                        help='input height')
    parser.add_argument('--width', default=640, type=int,
                        help='input width')
    parser.add_argument(
        '--image-dir', default='datasets/infrared_images/images/test',
        help='directory containing input images (searched recursively)')
    parser.add_argument(
        '--mean', default=[0.485, 0.456, 0.406],
        type=float, nargs=3, metavar=('R', 'G', 'B'),
        help='RGB normalization mean (input pixels are scaled to [0, 1])')
    parser.add_argument(
        '--std', default=[0.229, 0.224, 0.225],
        type=float, nargs=3, metavar=('R', 'G', 'B'),
        help='RGB normalization standard deviation')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='inference batch size')
    parser.add_argument('--warmup', default=50, type=int,
                        help='warmup iterations')
    parser.add_argument('--iterations', default=200, type=int,
                        help='timed batches per repeat')
    parser.add_argument('--repeats', default=3, type=int,
                        help='number of timed repeats')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='benchmark device')
    parser.add_argument('--fp16', action='store_true',
                        help='use FP16 inference (CUDA only)')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='optional .pth checkpoint')
    return parser.parse_args()


def validate_args(args):
    for name in ('num_classes', 'height', 'width', 'batch_size',
                 'iterations', 'repeats'):
        if getattr(args, name) <= 0:
            option = name.replace('_', '-')
            raise ValueError('--{} must be positive'.format(option))
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative')
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(
            'image directory not found: {}'.format(args.image_dir))
    if any(value <= 0 for value in args.std):
        raise ValueError('--std values must be positive')


def resolve_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(requested)


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint does not contain a state dict')
    for key in ('state_dict', 'model_state_dict', 'model'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_known_prefixes(key):
    prefixes = ('module.', 'model.', 'net.')
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'checkpoint not found: {}'.format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []

    for key, value in checkpoint.items():
        model_key = strip_known_prefixes(key)
        if model_key not in model_state:
            unexpected.append(key)
            continue
        if not hasattr(value, 'shape') or model_state[model_key].shape != value.shape:
            shape_mismatch.append(model_key)
            continue
        loaded[model_key] = value

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    missing = [key for key in model_state if key not in loaded]

    print('Checkpoint: {}'.format(checkpoint_path))
    print('Loaded state entries: {} / {}'.format(len(loaded), len(model_state)))
    if missing:
        print('Missing state entries: {}'.format(len(missing)))
    if unexpected:
        print('Unexpected state entries: {}'.format(len(unexpected)))
    if shape_mismatch:
        print('Shape-mismatched entries: {}'.format(shape_mismatch))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def find_image_files(image_dir):
    image_paths = []
    for directory, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(directory, filename))
    image_paths.sort()
    if not image_paths:
        raise RuntimeError(
            'no supported images found in {}'.format(image_dir))
    return image_paths


def preprocess_image(image_path, height, width, mean, std):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('failed to read image: {}'.format(image_path))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)[:, :, ::-1] / 255.0
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return np.ascontiguousarray(image)


def inference_with_postprocess(model, input_tensor):
    logits = model(input_tensor)
    return torch.argmax(logits, dim=1)


def select_batch(image_paths, batch_index, batch_size):
    start = batch_index * batch_size
    return [
        image_paths[(start + offset) % len(image_paths)]
        for offset in range(batch_size)]


def end_to_end_inference(model, batch_paths, height, width,
                         mean, std, dtype, device):
    images = [
        preprocess_image(path, height, width, mean, std)
        for path in batch_paths]
    cpu_batch = np.stack(images, axis=0)
    input_tensor = torch.from_numpy(cpu_batch).to(
        device=device, dtype=dtype, non_blocking=False)
    return inference_with_postprocess(model, input_tensor).cpu()


def benchmark(model, image_paths, args, mean, std, dtype, device):
    with torch.inference_mode():
        for batch_index in range(args.warmup):
            batch_paths = select_batch(
                image_paths, batch_index, args.batch_size)
            end_to_end_inference(
                model, batch_paths, args.height, args.width,
                mean, std, dtype, device)
        synchronize(device)

        results = []
        for repeat_index in range(args.repeats):
            synchronize(device)
            start = time.perf_counter()
            for batch_index in range(args.iterations):
                dataset_index = repeat_index * args.iterations + batch_index
                batch_paths = select_batch(
                    image_paths, dataset_index, args.batch_size)
                end_to_end_inference(
                    model, batch_paths, args.height, args.width,
                    mean, std, dtype, device)
            synchronize(device)
            elapsed = time.perf_counter() - start

            latency_ms = elapsed * 1000.0 / args.iterations
            fps = args.batch_size * args.iterations / elapsed
            results.append((latency_ms, fps))
    return results


def main():
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    if args.fp16 and device.type != 'cuda':
        raise ValueError('--fp16 is supported only with CUDA')

    torch.backends.cudnn.benchmark = device.type == 'cuda'
    model = SegFormer(
        num_classes=args.num_classes, phi=args.phi, pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model.half()

    image_paths = find_image_files(args.image_dir)
    mean = np.asarray(args.mean, dtype=np.float32).reshape((1, 1, 3))
    std = np.asarray(args.std, dtype=np.float32).reshape((1, 1, 3))

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)

    sample_paths = select_batch(image_paths, 0, args.batch_size)
    with torch.inference_mode():
        prediction = end_to_end_inference(
            model, sample_paths, args.height, args.width,
            mean, std, dtype, device)
    synchronize(device)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            'expected a tensor prediction, got {}'.format(type(prediction)))

    print('Model: SegFormer-{}'.format(args.phi.upper()))
    print('Device: {} ({})'.format(device, dtype))
    if device.type == 'cuda':
        print('GPU: {}'.format(torch.cuda.get_device_name(device)))
    print('Image directory: {}'.format(os.path.abspath(args.image_dir)))
    print('Images found: {}'.format(len(image_paths)))
    print('Input shape: {}'.format(
        (args.batch_size, 3, args.height, args.width)))
    print('Prediction shape: {}'.format(tuple(prediction.shape)))
    print('Timing scope: read/decode + resize/normalize + H2D + forward '
          '+ argmax + D2H')
    print('Parameters: {:,} ({:.6f} M)'.format(
        total_params, total_params / 1e6))
    print('Trainable parameters: {:,} ({:.6f} M)'.format(
        trainable_params, trainable_params / 1e6))
    print('Warmup / iterations / repeats: {} / {} / {}'.format(
        args.warmup, args.iterations, args.repeats))

    results = benchmark(
        model, image_paths, args, mean, std, dtype, device)
    latencies = [item[0] for item in results]
    fps_values = [item[1] for item in results]
    for index, (latency, fps) in enumerate(results, start=1):
        print('Repeat {}: latency={:.3f} ms/batch, FPS={:.3f} images/s'.format(
            index, latency, fps))

    print('Average latency: {:.3f} ms/batch'.format(
        statistics.mean(latencies)))
    print('Average FPS: {:.3f} images/s'.format(
        statistics.mean(fps_values)))
    if len(results) > 1:
        print('FPS std: {:.3f}'.format(statistics.stdev(fps_values)))


if __name__ == '__main__':
    main()
