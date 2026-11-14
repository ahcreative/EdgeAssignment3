"""
Shared CIFAR-10 dataset loader used by every script in this project.

Auto-detects and supports BOTH official distributions from
http://www.cs.toronto.edu/~kriz/cifar.html:

  * "binary version" (cifar-10-binary.tar.gz): data_batch_1.bin ... data_batch_5.bin,
    test_batch.bin. No header -- each file is a flat sequence of 10000 records of
    1 label byte + 3072 pixel bytes (1024 R, 1024 G, 1024 B, row-major 32x32).

  * "python version" (cifar-10-python.tar.gz, i.e. the cifar-10-batches-py folder):
    data_batch_1 ... data_batch_5, test_batch (no extension). Each file is a pickled
    dict with keys b'data' (10000x3072 uint8, same RGB-planar layout as above) and
    b'labels' (list of 10000 ints).

Point --data-dir (or `root=`) at whichever folder you actually have -- the class
figures out which format it is from the files present and loads accordingly.
"""
import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CIFAR10Bin(Dataset):
    RECORD_BYTES = 1 + 32 * 32 * 3  # 1 label byte + 3072 pixel bytes (binary format only)

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform

        bin_names = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
        pkl_names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]

        if all(os.path.exists(os.path.join(root, f)) for f in bin_names):
            self.data, self.targets = self._load_binary(root, bin_names)
        elif all(os.path.exists(os.path.join(root, f)) for f in pkl_names):
            self.data, self.targets = self._load_pickle(root, pkl_names)
        else:
            raise FileNotFoundError(
                f"Could not find CIFAR-10 files in {root}.\n"
                f"Expected either the binary-format files {bin_names}\n"
                f"or the python-format (pickle) files {pkl_names}.\n"
                f"Point --data-dir at the folder that directly contains one of these sets."
            )

    @staticmethod
    def _load_binary(root, files):
        images, labels = [], []
        for fname in files:
            path = os.path.join(root, fname)
            buf = np.fromfile(path, dtype=np.uint8)
            n = buf.size // CIFAR10Bin.RECORD_BYTES
            buf = buf[: n * CIFAR10Bin.RECORD_BYTES].reshape(n, CIFAR10Bin.RECORD_BYTES)
            labels.append(buf[:, 0])
            img = buf[:, 1:].reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)  # -> N,H,W,C uint8
            images.append(img)
        data = np.concatenate(images, axis=0)
        targets = np.concatenate(labels, axis=0).astype(np.int64)
        return data, targets

    @staticmethod
    def _load_pickle(root, files):
        images, labels = [], []
        for fname in files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            arr = batch[b"data"]  # (10000, 3072) uint8, same R/G/B-planar layout
            lbl = batch[b"labels"]
            images.append(arr.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))  # -> N,H,W,C
            labels.append(np.array(lbl, dtype=np.int64))
        data = np.concatenate(images, axis=0)
        targets = np.concatenate(labels, axis=0)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])  # HWC uint8 -> PIL, so torchvision transforms work
        label = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
