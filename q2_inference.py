"""
Q2 - Step 5: Unified inference pipeline for the original FP32 model, the unstructured-
pruned model, and the structured-pruned (reconstructed) model.

Usage:
    python q2_inference.py --data-dir <...> --model-type fp32 --ckpt resnet18_fp32.pth
    python q2_inference.py --data-dir <...> --model-type unstructured --ckpt resnet18_unstructured_pruned.pth
    python q2_inference.py --data-dir <...> --model-type structured --ckpt resnet18_structured_pruned_full.pth
"""
import argparse

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
    elif model_type == "structured":
        # architecture itself was reshaped -- load the full pickled model object
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(model_type)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, x):
    return model(x).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-type", required=True, choices=["fp32", "unstructured", "structured"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    test_set = CIFAR10Bin(args.data_dir, train=False, transform=tf)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = load_model(args.model_type, args.ckpt)
    print(f"Loaded {args.model_type} model from {args.ckpt}\n")

    shown = 0
    for x, y in loader:
        p = predict_batch(model, x).item()
        t = y.item()
        print(f"pred={CIFAR10_CLASSES[p]:<10} true={CIFAR10_CLASSES[t]:<10} {'OK' if p == t else 'X'}")
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()
