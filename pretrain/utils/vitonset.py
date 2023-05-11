import json
import os
from enum import IntEnum, Enum
from os.path import join, exists
from typing import Any, Callable, Tuple

import PIL.Image as PImage
from PIL import ImageDraw
import cv2
import numpy as np
import torch
from PIL.Image import Resampling
from numpy.linalg import lstsq
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


class ATRClass(IntEnum):
    """
    Deep Human Parsing with Active Template Regression (ATR) dataset classes
    """
    background = 0
    hat = 1
    hair = 2
    sunglasses = 3
    upper_clothes = 4
    skirt = 5
    pants = 6
    dress = 7
    belt = 8
    left_shoe = 9
    right_shoe = 10
    head = 11
    left_leg = 12
    right_leg = 13
    left_arm = 14
    right_arm = 15
    bag = 16
    scarf = 17


SHRUNK_ATR_LABELS = {
    0: ['background', [ATRClass.background]],
    1: ['hair', [ATRClass.hair, ATRClass.hat]],
    2: ['face', [ATRClass.head, ATRClass.sunglasses]],
    3: ['dress', [ATRClass.skirt, ATRClass.pants, ATRClass.dress, ATRClass.upper_clothes]],
    4: ['left_arm', [ATRClass.left_arm]],
    5: ['right_arm', [ATRClass.right_arm]],
    6: ['left_leg', [ATRClass.left_leg]],
    7: ['right_leg', [ATRClass.right_leg]],
    8: ['left_shoe', [ATRClass.left_shoe]],
    9: ['right_shoe', [ATRClass.right_shoe]],
    10: ['noise', [ATRClass.belt, ATRClass.bag, ATRClass.scarf]]
}


class ClothingType(Enum):
    """
    Defines different clothing types based on where these are worn
    """
    upper_body = 0
    lower_body = 1
    dresses = 2


CLOTH_EXTENDED_FIXED_MASK_LABELS_ATR = {
    ClothingType.dresses: [],
    ClothingType.upper_body: [ATRClass.skirt, ATRClass.pants],
    ClothingType.lower_body: [ATRClass.upper_clothes, ATRClass.left_arm, ATRClass.right_arm],
}


def pil_loader(ipath, mpath):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(ipath, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    with open(mpath, 'rb') as f:
        mask: PImage.Image = PImage.open(f).convert('L')
    return np.array(img), np.array(mask)


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def semantic_to_onehot(in_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros((num_classes, in_tensor.shape[0],
                        in_tensor.shape[1])).scatter_(0,
                                                      in_tensor.to(torch.long).unsqueeze(0), 1.0)


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


class VitonPoseDataset(Dataset):
    def __init__(
            self,
            root_folder: str,
            transform_img: Callable,
            shrink_labels: bool = False,
    ):
        self.root_folder = os.path.join(root_folder)
        self.transform_img = transform_img
        self.rgb_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.in_channels = (len(SHRUNK_ATR_LABELS) if shrink_labels else len(ATRClass)) + 3
        self.width = 192
        self.height = 256
        self.densepose_palette = get_palette(25)  # len(DenseposeClass)
        self.garment_type = ClothingType.dresses
        self.shrink_labels = shrink_labels

        # dresscode poses
        dress_denseposes = join(root_folder, 'dresses', 'dense_lowres')
        dress_denseposes = [join(dress_denseposes, x) for x in os.listdir(dress_denseposes)]
        dress_labels = [join(root_folder, 'dresses', 'label_maps_lowres', x.split('/')[-1].replace('_5.png', '_4.png')) for x in dress_denseposes]
        dress_keypoints = [join(root_folder, 'dresses', 'keypoints', x.split('/')[-1].replace('_5.png', '_2.json')) for x in dress_denseposes]

        self.densepose = dress_denseposes
        self.labels = dress_labels
        self.keypoints = dress_keypoints

        # dresscode upper body
        dress_denseposes = join(root_folder, 'upper_body', 'dense_lowres')
        dress_denseposes = [join(dress_denseposes, x) for x in os.listdir(dress_denseposes)]
        dress_labels = [join(root_folder, 'upper_body', 'label_maps_lowres', x.split('/')[-1].replace('_5.png', '_4.png'))
                        for x in dress_denseposes]
        dress_keypoints = [join(root_folder, 'upper_body', 'keypoints', x.split('/')[-1].replace('_5.png', '_2.json')) for
                           x in dress_denseposes]

        self.densepose += dress_denseposes
        self.labels += dress_labels
        self.keypoints += dress_keypoints

        # vitonhd upper body
        dress_denseposes = join(root_folder, 'viton_hd', 'dense_lowres')
        dress_denseposes = [join(dress_denseposes, x) for x in os.listdir(dress_denseposes)]
        dress_labels = [
            join(root_folder, 'viton_hd', 'label_maps_lowres', x.split('/')[-1].replace('_5.png', '_4.png'))
            for x in dress_denseposes]
        dress_keypoints = [join(root_folder, 'viton_hd', 'keypoints', x.split('/')[-1].replace('_5.png', '_2.json'))
                           for
                           x in dress_denseposes]

        self.densepose += dress_denseposes
        self.labels += dress_labels
        self.keypoints += dress_keypoints

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
            # orphan onfig
            root = join(root_folder, 'onfig_dense', b)
            denseposes = [join(root, x) for x in os.listdir(root)]
            labels = [join(root_folder, 'onfig_label_maps', b, x.split('/')[-1].replace('_5.png', '_4.png')) for x in denseposes]
            keypoints = [join(root_folder, 'onfig_keypoints', b, x.split('/')[-1].replace('_5.png', '_2.json')) for x in denseposes]

            self.densepose += denseposes
            self.labels += labels
            self.keypoints += keypoints

            # paired laydowns
            for f in ['_woman_dresses_faceless', '_woman_dresses_faces']:
                root = join(root_folder, b + f, 'dense_lowres')
                if not exists(root):
                    print('{} does not exist'.format(root))
                    continue
                denseposes = [join(root, x) for x in os.listdir(root)]
                labels = [join(root_folder, b + f, 'label_maps_lowres', x.split('/')[-1].replace('_5.png', '_4.png')) for x in denseposes]
                keypoints = [join(root_folder, b + f, 'keypoints', x.split('/')[-1].replace('_5.png', '_2.json'))
                          for x in denseposes]
                self.densepose += denseposes
                self.labels += labels
                self.keypoints += keypoints

        assert len(self.densepose) == len(self.labels) == len(self.keypoints)

    def draw_arms(self, pose_data):
        im_arms = PImage.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)

        shoulder_right = np.multiply(tuple(pose_data[2, :2]), self.height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5, :2]), self.height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3, :2]), self.height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6, :2]), self.height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4, :2]), self.height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7, :2]), self.height / 512.0)
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line(
                    np.concatenate((wrist_left, elbow_left, shoulder_left,
                                    shoulder_right)).astype(np.uint16).tolist(), 'white', 30,
                    'curve')
            else:
                arms_draw.line(
                    np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right,
                                    elbow_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line(
                    np.concatenate((shoulder_left, shoulder_right, elbow_right,
                                    wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
            else:
                arms_draw.line(
                    np.concatenate((elbow_left, shoulder_left, shoulder_right, elbow_right,
                                    wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
        else:
            arms_draw.line(
                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right,
                                wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')

        im_arms = np.array(im_arms)
        if self.height > 512:
            im_arms = cv2.dilate(im_arms, np.ones((10, 10), np.uint8), iterations=5)

        return im_arms

    def _remove_neck(self, head_mask, pose_data, label_map):
        head_mask_2 = np.copy(head_mask)
        points = [
            np.multiply(tuple(pose_data[2, :2]), self.height / 512.0),  # right shoulder
            np.multiply(tuple(pose_data[5, :2]), self.height / 512.0),  # left shoulder
        ]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        for i in range(label_map.shape[1]):
            y = i * m + c
            head_mask_2[int(y - 20 * (self.height / 512.0)):, i] = 0
        return head_mask_2

    def __getitem__(self, index: int):
        dppath, lbpath, kppath = self.densepose[index], self.labels[index], self.keypoints[index]

        # Load semantic labels
        im_parse = PImage.open(lbpath).convert('P')
        im_parse = im_parse.resize((self.width, self.height), Resampling.NEAREST)  # PIL
        label_map = np.array(im_parse)

        # Load keypoints
        with open(kppath, 'r') as f:
            pose_label = json.load(f)
        pose_data = pose_label['keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 4))

        fixed_mask = np.isin(label_map, [
                ATRClass.hair, ATRClass.left_shoe, ATRClass.right_shoe, ATRClass.hat,
                ATRClass.sunglasses, ATRClass.scarf, ATRClass.bag
            ] + CLOTH_EXTENDED_FIXED_MASK_LABELS_ATR[self.garment_type])

        arms_mask = np.isin(label_map, [ATRClass.left_arm, ATRClass.right_arm])
        head_mask = np.isin(label_map, [ATRClass.hat, ATRClass.hair, ATRClass.sunglasses, ATRClass.head])

        if self.garment_type in [ClothingType.dresses, ClothingType.upper_body]:
            try:
                im_arms = self.draw_arms(pose_data).astype(np.bool)
                hands = np.logical_and(np.logical_not(im_arms), arms_mask)
            except:
                hands = arms_mask
            # cloth_limbs_mask |= im_arms
            fixed_mask |= hands

        # Delete neck
        if self.garment_type in [ClothingType.dresses, ClothingType.upper_body]:
            try:
                head_mask_without_neck = self._remove_neck(head_mask, pose_data, label_map)
            except:
                head_mask_without_neck = head_mask
        else:
            head_mask_without_neck = head_mask

        fixed_mask = np.logical_or(fixed_mask, head_mask_without_neck)

        if self.shrink_labels:
            num_classes = len(SHRUNK_ATR_LABELS)
            label_map_shrunk = np.zeros((label_map.shape[0], label_map.shape[1]))
            for k, v in SHRUNK_ATR_LABELS.items():
                label_map_shrunk += (np.isin(label_map, v[1]) * k)
            label_map = label_map_shrunk  # HxW
        else:
            num_classes = len(ATRClass)

        h = label_map * fixed_mask  # HxW (uint8)

        # Load densepose
        dp = PImage.open(dppath).convert('L')
        dp = dp.resize((self.width, self.height), Resampling.NEAREST)
        dp.putpalette(self.densepose_palette)
        dp = dp.convert('RGB')
        dp = np.array(dp)

        transformed_images = self.transform_img(image=h, dp=dp)
        dp = self.rgb_normalize(transformed_images['dp'].to(torch.float) / 255)
        h = semantic_to_onehot(transformed_images['image'][0], num_classes)

        result = torch.cat(
            [
                h,  # 18 - cloth agnostic semantic map (binary)
                dp,  # 3 RGB image of densepose map (-1..1)
            ],
            0)

        return result

    def __len__(self):
        return len(self.densepose)


def build_viton_cloths(root_folder, input_size):
    trans_train_img = A.Compose([
        A.RandomResizedCrop(256, 192, scale=(0.67, 1.0), interpolation=cv2.INTER_CUBIC),
        ToTensorV2(),
    ])
    trans_train_img.add_targets(additional_targets={'mask': 'image'})

    dataset_train = VitonClothDataset(root_folder=root_folder, transform_img=trans_train_img)
    print_transform(trans_train_img, '[pre-train]')
    return dataset_train


def build_viton_pose(root_folder, input_size, **kwargs):
    trans_train_img = A.Compose([
        A.RandomResizedCrop(256, 192, scale=(0.67, 1.0), interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ])
    trans_train_img.add_targets(additional_targets={'dp': 'image'})

    dataset_train = VitonPoseDataset(root_folder=root_folder, transform_img=trans_train_img, **kwargs)
    print_transform(trans_train_img, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
