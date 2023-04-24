import os
from os.path import join, exists
from typing import Any, Callable, Tuple

import PIL.Image as PImage
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A
from torchvision.transforms import transforms

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(ipath, mpath):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(ipath, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    with open(mpath, 'rb') as f:
        mask: PImage.Image = PImage.open(f).convert('L')
    return np.array(img), np.array(mask)


class VitonClothDataset(Dataset):
    def __init__(
            self,
            root_folder: str,
            transform_img: Callable,
    ):
        self.root_folder = os.path.join(root_folder)
        self.loader = pil_loader
        self.transform_img = transform_img
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.in_channels = 4

        def valid_dresscode_cloth(x):
            return x.lower().endswith('_1.jpg')

        # dresscode laydowns
        dress_images = join(root_folder, 'dresses', 'images_lowres')
        dress_images = [join(dress_images, x) for x in os.listdir(dress_images) if valid_dresscode_cloth(x)]
        dress_masks = [join(root_folder, 'dresses', 'cloth_masks_lowres', x.split('/')[-1].split('.')[-2] + '.png') for x in dress_images]

        self.images = dress_images
        self.masks = dress_masks

        # dresscode upper body
        dress_images = join(root_folder, 'upper_body', 'images_lowres')
        dress_images = [join(dress_images, x) for x in os.listdir(dress_images) if valid_dresscode_cloth(x)]
        dress_masks = [join(root_folder, 'upper_body', 'cloth_masks_lowres', x.split('/')[-1].split('.')[-2] + '.png') for
                       x in dress_images]

        self.images += dress_images
        self.masks += dress_masks

        # vitonhd upper body
        dress_images = join(root_folder, 'viton_hd', 'images_lowres')
        dress_images = [join(dress_images, x) for x in os.listdir(dress_images) if valid_dresscode_cloth(x)]
        dress_masks = [join(root_folder, 'viton_hd', 'cloth_masks_lowres', x.split('/')[-1].split('.')[-2] + '.png')
                       for
                       x in dress_images]

        self.images += dress_images
        self.masks += dress_masks

        # other brands + masks
        brands = [
            'farfetch',
            'gucci',
            'lv',
            'otto',
            'zalando',
            'zalora',
            'zara'
        ]

        for b in brands:
            # orphan laydowns
            root = join(root_folder, 'laydown', b)
            images = [join(root, x) for x in os.listdir(root) if valid_dresscode_cloth(x)]
            masks = [join(root_folder, 'laydown_masks', b, x.split('/')[-1].split('.')[-2] + '.png') for x in images]

            self.images += images
            self.masks += masks

            # paired laydowns
            for f in ['_woman_dresses_faceless', '_woman_dresses_faces']:
                root = join(root_folder, b + f, 'images_lowres')
                if not exists(root):
                    continue
                images = [join(root, x) for x in os.listdir(root) if valid_dresscode_cloth(x)]
                masks = [join(root_folder, b + f, 'cloth_masks_lowres', x.split('/')[-1].split('.')[-2] + '.png') for x in
                         images]
                self.images += images
                self.masks += masks

        assert len(self.images) == len(self.masks)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        ipath, mpath = self.images[index], self.masks[index]
        img, mask = self.loader(ipath, mpath)

        transformed_images = self.transform_img(image=img, mask=mask)
        # normalize
        img = self.normalize(transformed_images['image'].to(torch.float) / 255)
        mask = transformed_images['mask'].to(torch.float) / 255   # 0..1 float

        return torch.cat([img, mask], 0)

    def __len__(self):
        return len(self.images)


def build_viton_cloths(root_folder, input_size):
    trans_train_img = A.Compose([
        A.RandomResizedCrop(256, 192, scale=(0.67, 1.0), interpolation=cv2.INTER_CUBIC),
        ToTensorV2(),
    ])
    trans_train_img.add_targets(additional_targets={'mask': 'image'})

    dataset_train = VitonClothDataset(root_folder=root_folder, transform_img=trans_train_img)
    print_transform(trans_train_img, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
