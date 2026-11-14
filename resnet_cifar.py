"""
ResNet-18 adapted for CIFAR-10 (32x32 inputs): 3x3 stem, no stem max-pool, used by both
Q1 (quantization) and Q2 (pruning).

Quantization-readiness built in from the start (so the SAME architecture trains FP32,
then gets fused/PTQ'd/QAT'd in Q1 without any surgery):
  * QuantStub / DeQuantStub bookend the forward pass.
  * The residual add uses nn.quantized.FloatFunctional (required for a quantized add;
    behaves as a plain '+' in FP32 too, so there is no cost to including it here).
  * fuse_model() fuses every Conv+BN(+ReLU) for both Q1's PTQ and QAT paths.
"""
import torch
import torch.nn as nn
import torch.quantization as tq


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        return out

    def fuse_model(self):
        tq.fuse_modules(self, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
        if self.downsample is not None:
            tq.fuse_modules(self.downsample, [["0", "1"]], inplace=True)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse the stem Conv+BN+ReLU, then every block's internal fusions."""
        tq.fuse_modules(self, [["conv1", "bn1", "relu1"]], inplace=True)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.fuse_model()
