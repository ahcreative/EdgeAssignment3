"""
Q1 - Step 3: Quantization-Aware Training (QAT).

Pipeline:
    load trained FP32 checkpoint
    -> fuse Conv+BN(+ReLU)
    -> set the QAT qconfig and prepare_qat() (inserts FakeQuantize modules)
    -> fine-tune a few epochs at a SMALL learning rate so the network adapts to the
       quantization noise it now sees on every forward pass
    -> freeze BatchNorm running stats and then the observers themselves in the last
       epochs (standard QAT recipe -- stabilizes the scale/zero-point before convert)
    -> convert() to a true INT8 model

Usage:
    python q1_qat.py --data-dir /kaggle/input/<slug>/cifar-10-batches-bin \
                      --fp32-ckpt resnet18_fp32.pth --epochs 5
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cifar_bin_dataset import CIFAR10Bin
from resnet_cifar import ResNet18CIFAR

try:
    from torch.ao.nn.intrinsic.qat import freeze_bn_stats
except ImportError:  # older torch versions
    from torch.nn.intrinsic.qat import freeze_bn_stats

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


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
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--freeze-bn-after", type=int, default=3,
                     help="epoch index after which BN running stats are frozen")
    ap.add_argument("--freeze-observers-after", type=int, default=3,
                     help="epoch index after which fake-quant observers stop updating")
    args = ap.parse_args()

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = CIFAR10Bin(args.data_dir, train=True, transform=train_tf)
    test_set = CIFAR10Bin(args.data_dir, train=False, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = ResNet18CIFAR(num_classes=10)
    model.load_state_dict(torch.load(args.fp32_ckpt, map_location="cpu"))
    model.train()
    model.fuse_model()

    torch.backends.quantized.engine = args.backend
    model.qconfig = tq.get_default_qat_qconfig(args.backend)
    tq.prepare_qat(model, inplace=True)

    # Fake-quant ops train fine on GPU; only the final convert() step needs CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        if epoch >= args.freeze_bn_after:
            model.apply(freeze_bn_stats)
        if epoch >= args.freeze_observers_after:
            model.apply(tq.disable_observer)

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

        model.eval()
        acc = evaluate(model, test_loader, device)  # fake-quant accuracy: a good proxy for final INT8
        print(f"QAT epoch {epoch + 1}/{args.epochs} | loss {running_loss / len(train_loader.dataset):.4f} "
              f"| fake-quant test_acc {acc:.4f} | {time.time() - t0:.1f}s")
        best_acc = max(best_acc, acc)

    model.to("cpu").eval()
    quantized_model = tq.convert(model, inplace=False)
    quantized_model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "resnet18_qat_int8.pth")
    torch.save(quantized_model.state_dict(), out_path)
    torch.save(quantized_model, os.path.join(args.out_dir, "resnet18_qat_int8_full.pth"))
    print(f"\nBest fake-quant test accuracy during QAT fine-tuning: {best_acc:.4f}")
    print(f"Saved QAT INT8 state_dict to {out_path}")
    print("Reload with quant_model_builder.load_qat_model(ckpt_path, backend).")


if __name__ == "__main__":
    main()
