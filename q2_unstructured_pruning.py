"""
Q2 - Step 2: Iterative unstructured (magnitude) pruning of the pretrained ResNet-18,
using torch.nn.utils.prune, global L1 ranking across all Conv2d/Linear weights, with
fine-tuning between rounds and permanent masking (prune.remove) at the end.

Sparsity schedule: each round's TARGET TOTAL sparsity is converted into a "fraction of
the still-alive weights to remove this round" before calling prune.global_unstructured.
(Computing each round's threshold straight off the round's target sparsity compounds --
5 rounds aiming for 45/70/83/91/95% would actually land at 45/83/97/99.7/100% total.)

Usage:
    python q2_unstructured_pruning.py --data-dir <...> --fp32-ckpt resnet18_fp32.pth \
        --target-sparsity 0.8 --rounds 5 --finetune-epochs-per-round 3
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cifar_bin_dataset import CIFAR10Bin
from resnet_cifar import ResNet18CIFAR

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def prunable_params(model):
    """(module, 'weight') pairs for every Conv2d and Linear layer."""
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def global_sparsity(model):
    zeros = total = 0
    for m, name in prunable_params(model):
        w = getattr(m, name)
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / total


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--target-sparsity", type=float, default=0.8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--finetune-epochs-per-round", type=int, default=3)
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

    model = ResNet18CIFAR(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.fp32_ckpt, map_location=device))
    acc0 = evaluate(model, test_loader, device)
    print(f"Pretrained FP32 accuracy before pruning: {acc0:.4f}")

    params = prunable_params(model)

    schedule = [1 - (1 - args.target_sparsity) ** ((r + 1) / args.rounds) for r in range(args.rounds)]
    print(f"Sparsity schedule (total, per round): {[f'{s:.3f}' for s in schedule]}")

    for r, target_total in enumerate(schedule):
        cur = global_sparsity(model)
        frac_of_alive = max(0.0, (target_total - cur) / (1 - cur)) if cur < 1 else 0.0
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=frac_of_alive)
        actual = global_sparsity(model)
        print(f"Round {r + 1}/{args.rounds}: target_total={target_total:.3f} actual_total={actual:.3f}")
        finetune(model, train_loader, test_loader, device, args.finetune_epochs_per_round, args.lr)

    # make pruning permanent: bake the mask into the weight tensor, drop the reparam hooks
    for m, name in params:
        prune.remove(m, name)

    final_acc = evaluate(model, test_loader, device)
    final_sparsity = global_sparsity(model)
    print(f"\nFinal unstructured-pruned model: sparsity={final_sparsity:.4f} accuracy={final_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "resnet18_unstructured_pruned.pth")
    torch.save(model.state_dict(), out_path)
    torch.save(model, os.path.join(args.out_dir, "resnet18_unstructured_pruned_full.pth"))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
