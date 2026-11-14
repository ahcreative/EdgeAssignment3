"""
Q2 - Step 3: Structured (channel) pruning of the pretrained ResNet-18.

Strategy: for every BasicBlock, rank the "hidden" channels between conv1 and conv2 by
the L1 norm of their conv1 filters, and physically rebuild conv1/bn1/conv2 with only
the top-k channels kept. This hidden channel sits strictly INSIDE each residual block
(conv1 -> bn1 -> relu -> conv2 -> bn2 -> +identity), so shrinking it changes real
compute (fewer filters, smaller matmuls, smaller tensors on disk) WITHOUT touching the
block's input/output channel count -- which must stay fixed for the residual add and
for every other block's input shape. This keeps tensor dimensions valid everywhere
without needing a general cross-layer dependency solver, while still being genuine
channel pruning (the removed channels are gone, not masked to zero).

Usage:
    python q2_structured_pruning.py --data-dir <...> --fp32-ckpt resnet18_fp32.pth \
        --keep-ratio 0.5 --finetune-epochs 10
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cifar_bin_dataset import CIFAR10Bin
from resnet_cifar import ResNet18CIFAR, BasicBlock

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def prune_block_channels(block: BasicBlock, keep_ratio: float):
    conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2
    planes = conv1.out_channels
    num_keep = max(1, int(round(planes * keep_ratio)))
    if num_keep >= planes:
        return

    importance = conv1.weight.data.abs().sum(dim=(1, 2, 3))  # L1 norm per output filter
    keep_idx = torch.argsort(importance, descending=True)[:num_keep]
    keep_idx, _ = torch.sort(keep_idx)

    in_planes = conv1.in_channels
    new_conv1 = nn.Conv2d(in_planes, num_keep, conv1.kernel_size, conv1.stride,
                           conv1.padding, bias=(conv1.bias is not None))
    new_conv1.weight.data = conv1.weight.data[keep_idx].clone()

    new_bn1 = nn.BatchNorm2d(num_keep)
    new_bn1.weight.data = bn1.weight.data[keep_idx].clone()
    new_bn1.bias.data = bn1.bias.data[keep_idx].clone()
    new_bn1.running_mean = bn1.running_mean[keep_idx].clone()
    new_bn1.running_var = bn1.running_var[keep_idx].clone()

    out_planes = conv2.out_channels
    new_conv2 = nn.Conv2d(num_keep, out_planes, conv2.kernel_size, conv2.stride,
                           conv2.padding, bias=(conv2.bias is not None))
    new_conv2.weight.data = conv2.weight.data[:, keep_idx, :, :].clone()

    block.conv1, block.bn1, block.conv2 = new_conv1, new_bn1, new_conv2


def prune_model_channels(model: ResNet18CIFAR, keep_ratio: float):
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            prune_block_channels(block, keep_ratio)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def finetune(model, loader, test_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    acc = evaluate(model, test_loader, device)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        print(f"    fine-tune epoch {ep + 1}/{epochs} | test_acc {acc:.4f}")
    return acc


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--keep-ratio", type=float, default=0.5,
                     help="fraction of each block's hidden channels to KEEP")
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = CIFAR10Bin(args.data_dir, train=True, transform=train_tf)
    test_set = CIFAR10Bin(args.data_dir, train=False, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = ResNet18CIFAR(num_classes=10)
    model.load_state_dict(torch.load(args.fp32_ckpt, map_location="cpu"))
    model.to(device)
    params_before = count_params(model)
    acc_before = evaluate(model, test_loader, device)
    print(f"Before structured pruning: params={params_before:,} acc={acc_before:.4f}")

    prune_model_channels(model, args.keep_ratio)
    model.to(device)
    params_after = count_params(model)
    acc_pruned = evaluate(model, test_loader, device)
    print(f"After channel pruning (keep_ratio={args.keep_ratio}): params={params_after:,} "
          f"({100 * params_after / params_before:.1f}% of original) acc={acc_pruned:.4f} (pre-fine-tune)")

    final_acc = finetune(model, train_loader, test_loader, device, args.finetune_epochs, args.lr)
    print(f"\nFinal structured-pruned model: params={params_after:,} "
          f"({100 * params_after / params_before:.1f}% of original) accuracy={final_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    # Save the FULL model object, not just a state_dict: the ARCHITECTURE itself changed
    # (smaller conv/bn layers), so a plain state_dict can't be reloaded into a fresh
    # ResNet18CIFAR() without first re-applying the exact same channel cuts.
    out_path = os.path.join(args.out_dir, "resnet18_structured_pruned_full.pth")
    torch.save(model, out_path)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "resnet18_structured_pruned_state.pth"))
    print(f"Saved reconstructed structured-pruned model to {out_path}")
    print("(Load with torch.load(path, weights_only=False) -- the architecture is custom-shaped. "
          "The *_state.pth file is kept for reference only; it needs the exact same keep_idx "
          "reconstruction to load into a fresh skeleton, so prefer the full-model file for inference.)")


if __name__ == "__main__":
    main()
