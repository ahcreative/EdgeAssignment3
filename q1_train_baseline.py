"""
Q1 - Step 1: Train an FP32 baseline ResNet-18 on CIFAR-10 (read from the binary dataset
files you already downloaded).

Usage (Kaggle, GPU on):
    python q1_train_baseline.py --data-dir /kaggle/input/<slug>/cifar-10-batches-bin --epochs 40

Saves (to --out-dir):
    resnet18_fp32.pth       - state_dict of the best FP32 model (by test accuracy)
    resnet18_fp32_full.pth  - full model object at the end of training (convenience)
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cifar_bin_dataset import CIFAR10Bin
from resnet_cifar import ResNet18CIFAR

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_loaders(data_dir, batch_size=128, num_workers=2):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_set = CIFAR10Bin(data_dir, train=True, transform=train_tf)
    test_set = CIFAR10Bin(data_dir, train=False, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    model = ResNet18CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                           weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs} | loss {train_loss:.4f} | "
              f"test_acc {test_acc:.4f} | lr {scheduler.get_last_lr()[0]:.5f} | "
              f"{time.time() - t0:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "resnet18_fp32.pth"))

    print(f"\nBest FP32 test accuracy: {best_acc:.4f}")
    torch.save(model, os.path.join(args.out_dir, "resnet18_fp32_full.pth"))
    print(f"Saved best state_dict to {os.path.join(args.out_dir, 'resnet18_fp32.pth')}")
    print(f"Saved final full model to {os.path.join(args.out_dir, 'resnet18_fp32_full.pth')}")


if __name__ == "__main__":
    main()
