"""
Q1 - Step 5: Automated benchmark of FP32, PTQ-INT8, and QAT-INT8 models on the full
CIFAR-10 test set. Produces:
    q1_benchmark_results.csv          - per-image predictions for every model
    q1_benchmark_results_summary.csv  - one row per model: accuracy, size, latency

Usage:
    python q1_benchmark.py --data-dir <...> \
        --fp32-ckpt resnet18_fp32.pth --ptq-ckpt resnet18_ptq_int8.pth --qat-ckpt resnet18_qat_int8.pth
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cifar_bin_dataset import CIFAR10Bin, CIFAR10_CLASSES
from resnet_cifar import ResNet18CIFAR
from quant_model_builder import load_ptq_model, load_qat_model

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_model(model_type, ckpt_path, backend):
    if model_type == "fp32":
        model = ResNet18CIFAR(num_classes=10)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    elif model_type == "ptq":
        model = load_ptq_model(ckpt_path, backend)
    elif model_type == "qat":
        model = load_qat_model(ckpt_path, backend)
    else:
        raise ValueError(model_type)
    model.eval()
    return model


@torch.no_grad()
def run_model(model, loader, lat_loader, warmup=10, latency_runs=200):
    all_preds, all_labels = [], []
    for x, y in loader:
        preds = model(x).argmax(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    x1, _ = next(iter(lat_loader))
    for _ in range(warmup):
        model(x1)
    lat = np.empty(latency_runs)
    for i in range(latency_runs):
        t0 = time.perf_counter()
        model(x1)
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return acc, all_preds, all_labels, lat


def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--ptq-ckpt", default="resnet18_ptq_int8.pth")
    ap.add_argument("--qat-ckpt", default="resnet18_qat_int8.pth")
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    ap.add_argument("--out-csv", default="q1_benchmark_results.csv")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    test_set = CIFAR10Bin(args.data_dir, train=False, transform=tf)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    lat_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    configs = [
        ("fp32", args.fp32_ckpt),
        ("ptq", args.ptq_ckpt),
        ("qat", args.qat_ckpt),
    ]

    summary_rows = []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "index", "true_label", "pred_label", "correct"])

        for name, ckpt in configs:
            if not os.path.exists(ckpt):
                print(f"[skip] {name}: checkpoint not found at {ckpt}")
                continue
            model = load_model(name, ckpt, args.backend)
            acc, preds, labels, lat = run_model(model, loader, lat_loader, latency_runs=200)
            for i, (p, t) in enumerate(zip(preds, labels)):
                writer.writerow([name, i, CIFAR10_CLASSES[t], CIFAR10_CLASSES[p], int(p == t)])
            size_mb = model_size_mb(ckpt)
            row = dict(model=name, accuracy=acc, size_mb=size_mb,
                       lat_mean_ms=lat.mean(), lat_p50_ms=np.percentile(lat, 50),
                       lat_p95_ms=np.percentile(lat, 95))
            summary_rows.append(row)
            print(f"{name:>5} | acc={acc:.4f} | size={size_mb:7.2f} MB | "
                  f"lat mean/p50/p95 = {lat.mean():.3f}/{np.percentile(lat, 50):.3f}/"
                  f"{np.percentile(lat, 95):.3f} ms")

    summary_csv = args.out_csv.replace(".csv", "_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "size_mb",
                                                "lat_mean_ms", "lat_p50_ms", "lat_p95_ms"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSaved per-image predictions to {args.out_csv}")
    print(f"Saved model-level summary to {summary_csv}")


if __name__ == "__main__":
    main()
