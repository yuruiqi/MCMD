import torch
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
import nibabel as nib
from monai.transforms import *
import pandas as pd
import math
import json
import matplotlib.pyplot as plt

from Data.Preprocess import join_path, get_filename_from_dir
from ImageProcess.Operations import seg_to_mask, get_box
from ImageProcess.Analysis import get_biggest_slice


def get_data(dict):
    data = LoadNiftid(keys=['image', 'mask'])(dict)
    data['label'] = np.array(dict['label'])
    return data


def aug_in(data, augments, group):
    """
    data: {'image': (1, h, w, d), 'mask':(1, h, w, d), 'label':int}
    augments:
    group:
    """
    datas = []
    for i in range(group):
        data_aug = data.copy()
        if augments is not None:
            for augment in augments:
                data_aug = augment(data_aug)

        seg_aug = np.where(data_aug['mask'] > 0.5, 1, 0)  # (1, h, w, d)
        data_aug['mask'] = seg_aug  # (1, h, w, d)
        datas.append(data_aug)
    data['image'] = np.concatenate([x['image'] for x in datas])  # (n, h, w, d)
    data['mask'] = np.concatenate([x['mask'] for x in datas])  # (n, h, w, d)
    return data


class MyDataset(Dataset):
    def __init__(self, datapath, config, preload=False, augment=False):
        self.shape = config['SHAPE']
        self.group = config['GROUP']
        self.mode = config['OUT MODE']

        with open(datapath) as f:
            self.datalist = json.load(f)  # [{'image':path, 'mask':path, 'label':int}, ]

        if preload:
            self.data = [get_data(dict) for dict in self.datalist]
        else:
            self.data = None

        pixdim = [0.7, 0.7, 1.25] if len(self.shape) == 3 else [0.7, 0.7]
        self.transforms = [AddChanneld(keys=['image', 'mask', 'label']),
                           Spacingd(keys=['image', 'mask'], pixdim=pixdim, mode=['bilinear', 'nearest']),
                           ResizeWithPadOrCropd(keys=['image', 'mask'], spatial_size=self.shape),
                           ]

        if augment:
            if self.mode == 'origin' or self.mode == 'all':
                self.augments = [RandFlipd(keys=['image', 'mask'], spatial_axis=2 if len(self.shape)==3 else None, prob=0.5),
                                 RandRotated(keys=['image', 'mask'], range_z=30 if len(self.shape)==3 else 0.0, prob=0.5),
                                 RandGaussianNoised(keys=['image'], prob=0.5),
                                 RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.5)
                                 ]
            elif self.mode == 'conservative':
                self.augments = [RandGaussianNoised(keys=['image'], prob=0.5),
                                 RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.5)
                                 ]
            else:
                raise ValueError
        else:
            self.augments = None

    def __getitem__(self, index):
        # index = 1  # For debug
        if self.data is None:
            dict = self.datalist[index]
            data = get_data(dict)
        else:
            data = self.data[index]

        # trans 2d if need
        if len(self.shape)==2:
            slice = get_biggest_slice(data['mask'])
            data['image'] = data['image'][:,:,slice]
            data['mask'] = data['mask'][:,:,slice]

        # print('before',data['image_meta_dict']['original_affine'])

        for transform in self.transforms:
            data = transform(data)
        # print('after', data['image_meta_dict']['affine'])

        if self.mode == 'origin':
            if self.augments is not None:
                for augment in self.augments:
                    data = augment(data)
        elif self.mode == 'all' or self.mode == 'conservative':
            data = aug_in(data, self.augments, self.group)

        img = data['image'].astype(np.float32)
        seg = data['mask'].astype(np.float32)
        label = data['label'].astype(np.float32)
        return img, seg, label

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    from Config import configs

    config = configs[4]
    dataset = MyDataset(config['TRAIN'], config, augment=True, preload=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for img, seg, label in dataloader:
        print(img.shape, seg.shape, label.shape)

        for i in range(4):
            image = img[0,i]
            segmentation = seg[0, i]
            slice = get_biggest_slice(segmentation)

            plt.imshow(image[:, :, slice], cmap='gray')
            plt.contour(segmentation[:, :, slice])
            plt.show()

        break
        # pass
