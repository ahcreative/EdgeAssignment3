"""
Q1 - Step 4: Unified inference pipeline supporting FP32, PTQ-INT8, and QAT-INT8 models.

Usage:
    python q1_inference.py --data-dir <...> --model-type fp32 --ckpt resnet18_fp32.pth
    python q1_inference.py --data-dir <...> --model-type ptq  --ckpt resnet18_ptq_int8.pth
    python q1_inference.py --data-dir <...> --model-type qat  --ckpt resnet18_qat_int8.pth
"""
import argparse

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cifar_bin_dataset import CIFAR10Bin, CIFAR10_CLASSES
from resnet_cifar import ResNet18CIFAR
from quant_model_builder import load_ptq_model, load_qat_model

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_model(model_type, ckpt_path, backend="fbgemm"):
    if model_type == "fp32":
        model = ResNet18CIFAR(num_classes=10)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        return model
    elif model_type == "ptq":
        return load_ptq_model(ckpt_path, backend)
    elif model_type == "qat":
        return load_qat_model(ckpt_path, backend)
    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def predict_batch(model, x):
    return model(x).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-type", required=True, choices=["fp32", "ptq", "qat"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=20, help="number of test images to print")
    args = ap.parse_args()

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    test_set = CIFAR10Bin(args.data_dir, train=False, transform=tf)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_type, args.ckpt, args.backend)
    print(f"Loaded {args.model_type} model from {args.ckpt}\n")

    shown = 0
    for x, y in loader:
        preds = predict_batch(model, x)
        for p, t in zip(preds.tolist(), y.tolist()):
            print(f"pred={CIFAR10_CLASSES[p]:<10} true={CIFAR10_CLASSES[t]:<10} "
                  f"{'OK' if p == t else 'X'}")
            shown += 1
            if shown >= args.limit:
                return


if __name__ == "__main__":
    main()
