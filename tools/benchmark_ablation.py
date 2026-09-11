#!/usr/bin/env python3
"""Benchmark SegFormer under one fixed protocol.

Fixed protocol
--------------
* Default model: SegFormer baseline
* Input: 1 x 3 x 640 x 640, FP32
* Device: the first visible CUDA device (use CUDA_VISIBLE_DEVICES to select it)
* Warm-up: 100 forward passes
* Timing: 500 forward passes, batch size 1
* Synchronization: torch.cuda.synchronize() immediately before and after every
  timed forward pass
* Complexity: THOP MACs converted with 1 MAC = 2 FLOPs
* Output: benchmark_results.csv

Checkpoint paths are optional because parameters and FLOPs depend on the model
structure rather than trained values. If supplied, checkpoints are loaded
before latency/FPS measurement.

Examples:
    python tools/benchmark_ablation.py

    python tools/benchmark_ablation.py \
        --baseline-checkpoint path/to/baseline.pth
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.segformer import SegFormer  # noqa: E402


INPUT_SHAPE = (1, 3, 640, 640)
NUM_CLASSES = 10
PHI = "b0"
WARMUP_RUNS = 100
TIMED_RUNS = 500
CSV_PATH = Path("benchmark_results.csv")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_coordatt: bool
    checkpoint_arg: str


# Only the plain baseline is benchmarked by default.
MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("SegFormer-B0 baseline", False, "baseline_checkpoint"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SegFormer-B0 baseline. Input "
            "size, batch size, precision, warm-up count, timed runs and CSV "
            "path are fixed."
        )
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        help="optional checkpoint for the baseline model",
    )
    return parser.parse_args()


def build_model(spec: ModelSpec) -> nn.Module:
    """Build a model and remove CoordAtt for the plain baseline."""
    model = SegFormer(num_classes=NUM_CLASSES, phi=PHI, pretrained=False)
    if not spec.use_coordatt:
        model.decode_head.coord_att = nn.Identity()
    return model


def unwrap_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping.")

    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = state.get(key)
        if isinstance(value, dict):
            state = value
            break

    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("No valid model state_dict was found in the checkpoint.")
    return state


def key_candidates(key: str) -> List[str]:
    """Return key variants for raw and wrapped checkpoints."""
    candidates = [key]
    current = key
    prefixes = ("module.", "model.", "net.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                candidates.append(current)
                changed = True
                break
    return candidates


def load_checkpoint(model: nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))

    raw_checkpoint = torch.load(str(path), map_location="cpu")
    raw_state = unwrap_state_dict(raw_checkpoint)
    model_state = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}

    for raw_key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for candidate in key_candidates(raw_key):
            if candidate in model_state and value.shape == model_state[candidate].shape:
                matched[candidate] = value
                break

    if not matched:
        raise RuntimeError(
            "No checkpoint tensors matched the model for: {}".format(path)
        )

    incompatible = model.load_state_dict(matched, strict=False)
    print(
        "  Loaded checkpoint: {} (matched {}, missing {}, unexpected {})".format(
            path,
            len(matched),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    )


def count_complexity(
    model: nn.Module, input_tensor: torch.Tensor
) -> Tuple[float, float]:
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            "THOP is required for complexity measurement. Install it with: "
            "pip install thop"
        ) from exc

    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    # THOP reports MACs. This follows the reference convention where one MAC
    # is one multiplication plus one addition, hence 1 MAC = 2 FLOPs.
    raw_macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    gflops = 2.0 * float(raw_macs) / 1e9
    return params_m, gflops


def measure_latency(
    model: nn.Module, input_tensor: torch.Tensor
) -> Tuple[float, float]:
    model.eval()
    elapsed_seconds: List[float] = []

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)
        torch.cuda.synchronize()

        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(input_tensor)
            torch.cuda.synchronize()
            elapsed_seconds.append(time.perf_counter() - start)

    latency_ms = sum(elapsed_seconds) / TIMED_RUNS * 1000.0
    fps = INPUT_SHAPE[0] * 1000.0 / latency_ms
    return latency_ms, fps


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = ("Model", "Params(M)", "GFLOPs", "Latency(ms)", "FPS")
    formatted = [
        (
            str(row["Model"]),
            "{:.4f}".format(row["Params(M)"]),
            "{:.3f}".format(row["GFLOPs"]),
            "{:.3f}".format(row["Latency(ms)"]),
            "{:.2f}".format(row["FPS"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in formatted))
        for index in range(len(headers))
    ]

    def line(values: Tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print("\n" + line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(line(row))


def save_csv(rows: List[Dict[str, object]]) -> None:
    fieldnames = ["Model", "Params(M)", "GFLOPs", "Latency(ms)", "FPS"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "Params(M)": "{:.4f}".format(row["Params(M)"]),
                    "GFLOPs": "{:.6f}".format(row["GFLOPs"]),
                    "Latency(ms)": "{:.6f}".format(row["Latency(ms)"]),
                    "FPS": "{:.6f}".format(row["FPS"]),
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for latency and FPS testing.")

    torch.manual_seed(304)
    torch.cuda.manual_seed_all(304)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    input_tensor = torch.randn(INPUT_SHAPE, device=device, dtype=torch.float32)

    print("PyTorch: {}".format(torch.__version__))
    print("CUDA runtime: {}".format(torch.version.cuda))
    print("GPU: {}".format(torch.cuda.get_device_name(device)))
    print("Input: {}, FP32, batch size 1".format(INPUT_SHAPE))
    print("Warm-up: {}; timed runs: {}".format(WARMUP_RUNS, TIMED_RUNS))
    print("GFLOPs: THOP MACs x 2 (1 MAC = 2 FLOPs)")

    results: List[Dict[str, object]] = []
    for index, spec in enumerate(MODEL_SPECS, start=1):
        print("\n[{}/{}] {}".format(index, len(MODEL_SPECS), spec.name))
        model = build_model(spec)

        checkpoint: Optional[Path] = getattr(args, spec.checkpoint_arg)
        if checkpoint is not None:
            load_checkpoint(model, checkpoint)
        else:
            print("  No checkpoint supplied; using initialized weights.")

        model.eval().to(device)
        params_m, gflops = count_complexity(model, input_tensor)
        latency_ms, fps = measure_latency(model, input_tensor)
        results.append(
            {
                "Model": spec.name,
                "Params(M)": params_m,
                "GFLOPs": gflops,
                "Latency(ms)": latency_ms,
                "FPS": fps,
            }
        )

        del model
        torch.cuda.empty_cache()

    print_table(results)
    save_csv(results)
    print("\nSaved CSV to: {}".format(CSV_PATH.resolve()))


if __name__ == "__main__":
    main()
