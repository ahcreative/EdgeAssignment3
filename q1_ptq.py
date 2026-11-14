"""
Q1 - Step 2: Post-Training Static Quantization (PTQ).

Pipeline:
    load trained FP32 checkpoint
    -> fuse Conv+BN(+ReLU)                    (model.fuse_model())
    -> configure the quantization backend     (fbgemm for x86 Kaggle CPUs; qnnpack for ARM)
    -> prepare() inserts observers
    -> calibrate on a subset of TRAIN data (forward passes only, no labels/backward)
    -> convert() turns the observed FP32 model into a real INT8 model
    -> save the quantized state_dict for deployment

Usage:
    python q1_ptq.py --data-dir /kaggle/input/<slug>/cifar-10-batches-bin \
                      --fp32-ckpt resnet18_fp32.pth
"""
import argparse
import os

import torch
import torch.quantization as tq
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from cifar_bin_dataset import CIFAR10Bin
from resnet_cifar import ResNet18CIFAR

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def calibrate(model, loader, num_batches):
    model.eval()
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"],
                     help="fbgemm for x86 CPUs (typical Kaggle CPU), qnnpack for ARM")
    ap.add_argument("--calib-batches", type=int, default=50)
    ap.add_argument("--calib-batch-size", type=int, default=64)
    args = ap.parse_args()

    # Eager-mode INT8 kernels are CPU-only -- PTQ/convert must run on CPU.
    device = torch.device("cpu")

    calib_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = CIFAR10Bin(args.data_dir, train=True, transform=calib_tf)
    n_calib = min(len(train_set), args.calib_batches * args.calib_batch_size)
    calib_subset = Subset(train_set, list(range(n_calib)))
    calib_loader = DataLoader(calib_subset, batch_size=args.calib_batch_size, shuffle=False)

    model = ResNet18CIFAR(num_classes=10)
    model.load_state_dict(torch.load(args.fp32_ckpt, map_location=device))
    model.to(device).eval()

    model.fuse_model()

    torch.backends.quantized.engine = args.backend
    model.qconfig = tq.get_default_qconfig(args.backend)
    print("Using qconfig:", model.qconfig)

    tq.prepare(model, inplace=True)
    print(f"Calibrating on {n_calib} training images ({args.calib_batches} batches)...")
    calibrate(model, calib_loader, args.calib_batches)

    tq.convert(model, inplace=True)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "resnet18_ptq_int8.pth")
    torch.save(model.state_dict(), out_path)
    torch.save(model, os.path.join(args.out_dir, "resnet18_ptq_int8_full.pth"))
    print(f"Saved PTQ INT8 state_dict to {out_path}")
    print(f"Saved PTQ INT8 full model to {os.path.join(args.out_dir, 'resnet18_ptq_int8_full.pth')}")
    print("Reload with quant_model_builder.load_ptq_model(ckpt_path, backend) "
          "(a quantized skeleton must be rebuilt before load_state_dict will work).")


if __name__ == "__main__":
    main()
