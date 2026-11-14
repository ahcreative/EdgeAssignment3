"""
Eager-mode quantized modules (quantized::conv2d, etc.) only exist on a model AFTER
prepare()/prepare_qat() + convert() has actually been run on it once. So to reload a
saved INT8 state_dict, we must first replay that exact recipe on a freshly constructed
FP32 model (fuse -> set qconfig -> prepare[_qat] -> convert) to get a skeleton with the
right quantized module types and shapes, and ONLY THEN call load_state_dict. This
module centralizes that so q1_inference.py and q1_benchmark.py don't duplicate it.
"""
import torch
import torch.quantization as tq
from resnet_cifar import ResNet18CIFAR


def build_quantized_skeleton(backend="fbgemm", qat=False):
    torch.backends.quantized.engine = backend
    model = ResNet18CIFAR(num_classes=10)
    model.eval()
    model.fuse_model()
    model.qconfig = tq.get_default_qat_qconfig(backend) if qat else tq.get_default_qconfig(backend)
    if qat:
        tq.prepare_qat(model, inplace=True)
    else:
        tq.prepare(model, inplace=True)
    model = tq.convert(model, inplace=False)
    model.eval()
    return model


def load_ptq_model(ckpt_path, backend="fbgemm"):
    model = build_quantized_skeleton(backend, qat=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def load_qat_model(ckpt_path, backend="fbgemm"):
    model = build_quantized_skeleton(backend, qat=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model
