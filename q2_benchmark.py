"""
Q2 - Step 6: Automated benchmark of the original, unstructured-pruned, and
structured-pruned models. Produces:
    q2_benchmark_results.csv          - per-image predictions for every model
    q2_benchmark_results_summary.csv  - one row per model: accuracy, params, size, latency

Usage:
    python q2_benchmark.py --data-dir <...> \
        --fp32-ckpt resnet18_fp32.pth \
        --unstructured-ckpt resnet18_unstructured_pruned.pth \
        --structured-ckpt resnet18_structured_pruned_full.pth
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

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_model(model_type, ckpt_path):
    if model_type in ("fp32", "unstructured"):
        model = ResNet18CIFAR(num_classes=10)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--unstructured-ckpt", default="resnet18_unstructured_pruned.pth")
    ap.add_argument("--structured-ckpt", default="resnet18_structured_pruned_full.pth")
    ap.add_argument("--out-csv", default="q2_benchmark_results.csv")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    test_set = CIFAR10Bin(args.data_dir, train=False, transform=tf)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    lat_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    configs = [
        ("fp32", args.fp32_ckpt),
        ("unstructured", args.unstructured_ckpt),
        ("structured", args.structured_ckpt),
    ]

    summary_rows = []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "index", "true_label", "pred_label", "correct"])
        for name, ckpt in configs:
            if not os.path.exists(ckpt):
                print(f"[skip] {name}: checkpoint not found at {ckpt}")
                continue
            model = load_model(name, ckpt)
            acc, preds, labels, lat = run_model(model, loader, lat_loader)
            for i, (p, t) in enumerate(zip(preds, labels)):
                writer.writerow([name, i, CIFAR10_CLASSES[t], CIFAR10_CLASSES[p], int(p == t)])
            n_params = count_params(model)
            n_nonzero = sum(int((p != 0).sum().item()) for p in model.parameters())
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            row = dict(model=name, accuracy=acc, total_params=n_params, nonzero_params=n_nonzero,
                       size_mb=size_mb, lat_mean_ms=lat.mean(), lat_p50_ms=np.percentile(lat, 50),
                       lat_p95_ms=np.percentile(lat, 95))
            summary_rows.append(row)
            print(f"{name:>12} | acc={acc:.4f} | total_params={n_params:,} | "
                  f"nonzero_params={n_nonzero:,} | size={size_mb:7.2f} MB | "
                  f"lat mean/p50/p95={lat.mean():.3f}/{np.percentile(lat, 50):.3f}/"
                  f"{np.percentile(lat, 95):.3f} ms")

    summary_csv = args.out_csv.replace(".csv", "_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "total_params", "nonzero_params",
                                                "size_mb", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSaved per-image predictions to {args.out_csv}")
    print(f"Saved model-level summary to {summary_csv}")
    print("\nNote: unstructured pruning zeroes weights in a still-DENSE tensor, so CPU latency "
          "typically will NOT drop (sparsity != speed -- a dense kernel still multiplies the "
          "zeros) unless run through a sparse-aware kernel/runtime. Structured (channel) pruning "
          "actually shrinks the tensors, so its latency and size improvements are real and should "
          "show up directly in this table.")


if __name__ == "__main__":
    main()
