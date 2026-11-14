"""
Q1 - ONNX deployment.

Exports the FP32 model to standard ONNX and shows how to run it with onnxruntime on CPU.

NOTE on INT8 ONNX export (read before you report "done" on this deliverable):
PyTorch eager-mode quantization produces custom `quantized::*` ops (quantized::conv2d,
etc.) that are PyTorch-internal -- torch.onnx.export does not have a built-in, supported
mapping from them to standard ONNX QOperator/QDQ ops, so directly exporting the
resnet18_ptq_int8 / resnet18_qat_int8 models does not give a working ONNX INT8 graph.
The officially supported route to a real INT8 ONNX model is the one used below:
    1. export the FP32 model to ONNX (export_fp32, below), then
    2. hand that FP32 ONNX graph to ONNX Runtime's OWN static quantizer
       (onnxruntime.quantization.quantize_static), calibrated the same way as the
       PyTorch PTQ pipeline -- a representative subset of training images.
This keeps the ONNX artifact fully portable (no dependency on PyTorch's internal
quantized op set) and is the standard pattern for PyTorch -> ONNX Runtime INT8 deployment.

Usage:
    python q1_export_onnx.py --fp32-ckpt resnet18_fp32.pth --out-dir .
    # then, to also produce an INT8 ONNX model:
    python q1_export_onnx.py --fp32-ckpt resnet18_fp32.pth --out-dir . \
        --data-dir /kaggle/input/<slug>/cifar-10-batches-bin --quantize-onnx
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from resnet_cifar import ResNet18CIFAR

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def export_fp32(ckpt_path, out_path):
    model = ResNet18CIFAR(num_classes=10)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported FP32 ONNX model to {out_path}")
    return out_path


class CIFARCalibrationReader:
    """onnxruntime CalibrationDataReader over a subset of CIFAR-10 training images,
    normalized exactly like the PyTorch pipeline so scales/zero-points line up."""

    def __init__(self, data_dir, input_name, n_samples=2000, batch_size=1):
        from cifar_bin_dataset import CIFAR10Bin
        import torchvision.transforms as T

        tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        self.dataset = CIFAR10Bin(data_dir, train=True, transform=tf)
        self.n_samples = min(n_samples, len(self.dataset))
        self.input_name = input_name
        self.batch_size = batch_size
        self._idx = 0

    def get_next(self):
        if self._idx >= self.n_samples:
            return None
        x, _ = self.dataset[self._idx]
        self._idx += 1
        return {self.input_name: x.unsqueeze(0).numpy().astype(np.float32)}

    def rewind(self):
        self._idx = 0


def quantize_onnx_static(fp32_onnx_path, data_dir, out_path, n_calib_samples=2000):
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

    sess = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = CIFARCalibrationReader(data_dir, input_name, n_samples=n_calib_samples)

    quantize_static(
        fp32_onnx_path, out_path, reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Exported statically-quantized INT8 ONNX model to {out_path}")
    return out_path


def run_onnx_inference(onnx_path, x_numpy):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: x_numpy.astype(np.float32)})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-ckpt", default="resnet18_fp32.pth")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--quantize-onnx", action="store_true",
                     help="also produce an INT8 ONNX model via onnxruntime static quantization")
    ap.add_argument("--data-dir", default=None, help="required if --quantize-onnx is set")
    ap.add_argument("--calib-samples", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fp32_onnx = export_fp32(args.fp32_ckpt, os.path.join(args.out_dir, "resnet18_fp32.onnx"))

    # sanity check: run it once with onnxruntime CPU EP
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
    out = run_onnx_inference(fp32_onnx, dummy)
    print(f"Sanity check: onnxruntime output shape {out.shape}")

    if args.quantize_onnx:
        if not args.data_dir:
            raise SystemExit("--quantize-onnx requires --data-dir for calibration")
        int8_onnx = os.path.join(args.out_dir, "resnet18_int8.onnx")
        quantize_onnx_static(fp32_onnx, args.data_dir, int8_onnx, args.calib_samples)
    else:
        print("\n(Skipping INT8 ONNX export -- re-run with --quantize-onnx --data-dir <...> "
              "to also produce resnet18_int8.onnx.)")


if __name__ == "__main__":
    main()
