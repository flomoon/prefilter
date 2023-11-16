from absl import logging

from pathlib import Path
from PIL import Image

import random
import numpy as np
import albumentations as A

from torch.utils.data import Dataset


def transform_fn(x, resizing_size):
    x = np.array(x, 'float32')                                    # PIL.Image -> np.float32
    x = A.Resize(resizing_size, resizing_size)(image=x)['image']  # resize (256, 256)
    x = x / 255.                                                  # normalize to [0, 1]
    x = x.transpose([2, 0, 1])                                    # (H, W , C) -> (C, H, W)
    return x


class ImageNet1K(Dataset):
    def __init__(self, root, split, transform=None):
        assert split in ['train', 'val']
        data_path = Path(root) / split
        self._files = sorted(data_path.glob('*/*.JPEG'))
        random.shuffle(self._files)
        classes = sorted([path.name for path in data_path.glob('*') if path.is_dir()])
        self._class_to_label = {cls_: i for i, cls_ in enumerate(classes)}
        self._transform = transform
        logging.info(f"ImageNet-1K '{split}' set is prepared.")
    
    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        img_file = self._files[idx]
        img = Image.open(img_file).convert('RGB')
        if self._transform:
            img = self._transform(img)

        img_class = img_file.parent.name
        img_label = self._class_to_label[img_class]
        return img, img_label
