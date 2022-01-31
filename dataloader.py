#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:38:36 2021


@author: Luca Medeiros, lucamedeiros@outlook.com
"""
import numpy as np
import torch
import os
import torchvision.transforms as transforms
import pytorch_lightning as pl

from torchvision import datasets
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset

from augmentations import weak_transform, strong_transform
import json
from typing import Any
from torchvision import get_image_backend
from PIL import Image
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import random

# loader, classes, samples
class ImageInstance(Dataset):
    def __init__(self, root_path, json_path, transform, batch_size):
        super(ImageInstance, self).__init__()
        self.transform = transform
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.samples = []
        for i, image in enumerate(json_data['images']):
            path = os.path.join(root_path, image['file_name'])
            target = image['category_id']
            self.samples.append((path, target))
        self.classes = []
        for category in json_data['categories']:
            self.classes.append(category['label'])
            
        self.normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                               (0.2023, 0.1994, 0.2010))

        self.ToTensor = transforms.ToTensor()
        self.t = transforms.Compose([
            transforms.Resize(224),             # resize shortest side to 224 pixels
            transforms.CenterCrop(224),         # crop longest side to 224 pixels at center,
        ])
    
    def pil_loader(self, path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    # TODO: specify the return type
    def accimage_loader(self, path: str) -> Any:
        try:
            return self.accimage.Image(path)
        except IOError:
            # Potentially a decoding pr
            return self.pil_loader(path)
        
        
    def loader(self, path: str) -> Any:
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)
        
    # def distibute_sampling(self):
        
    
    
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        basename = os.path.basename(path)
        class_name = self.classes[target]
        sample = self.loader(path)
        sample = self.t(sample)
        sample = np.array(sample, dtype=np.uint8)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.transform is not None:
            if isinstance(self.transform, tuple):
                w_sample = self.transform[0](image=sample)['image']
                s_sample = self.transform[1](image=sample)['image']
                w_sample = self.normalizer(self.ToTensor(w_sample))
                s_sample = self.normalizer(self.ToTensor(s_sample))
                return (w_sample.float(), s_sample.float()), target, index, basename, class_name
            else:
                sample = self.transform(image=sample)
                sample = sample['image']
        sample = self.normalizer(self.ToTensor(sample))
        return sample.float(), target, index, basename, class_name


class MyBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.oversampled = False

        self.batch_size = args.b
        transforms = None
        if self.args.augmentation == 'strong':
            transforms = strong_transform
        elif self.args.augmentation == 'weak':
            transforms = weak_transform
        self.transforms_train = transforms
        self.transforms_val = None

    def setup(self):
        # Called on every GPU
        self.train = ImageInstance(self.args.train_path, self.args.train_json,
                                   transform=self.transforms_train, batch_size=self.batch_size)
            
        self.classes = self.train.classes
        self.ndata = self.train.__len__()

        self.val = ImageInstance(self.args.valid_path, self.args.valid_json,
                                 transform=self.transforms_val, batch_size=self.batch_size)
        self.val_classes = self.val.classes
        # assert len(self.classes) == len(self.val.classes), 'Train {len(self.classes)} and val {len(self.val.classes)} classes doesnt match.'

    def oversample(self, dataset):
        smt = RandomOverSampler()
        tset = np.asarray(dataset.samples)
        print('Before oversampling:', len(tset))
        x, y = smt.fit_resample(tset[:, 0].reshape(-1, 1), tset[:, 1])
        print('After oversampling:', len(x))
        dataset.samples = [(str(dx[0]), int(dy)) for dx, dy in zip(x, y)]
        self.oversampled = True

    def train_dataloader(self):
        if self.args.oversample and not self.oversampled:
            self.oversample(self.train)
        class_idxs = []
        for i, cl in enumerate(self.classes):
            class_idxs.append([])
        for i, (image, targets) in enumerate(self.train.samples):
            class_idxs[targets].append(i)
        for class_idx in class_idxs:
            random.shuffle(class_idx)
            
        batch_index = []
        for i in range(len(self.train.samples)//self.batch_size):
            batch_index.append([])
            for j in range(self.batch_size):
                step = i * self.batch_size + j
                batch_index[-1].append(class_idxs[step%len(class_idxs)][step//len(class_idxs)])
        
        
        return torch.utils.data.DataLoader(self.train,
                                            # batch_size=self.batch_size,
                                            # shuffle=True,
                                            batch_sampler=MyBatchSampler(batch_index),
                                            num_workers=4)

    def val_dataloader(self, shuffle=False):
        print('Testset length: ', len(self.val))
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=self.batch_size,
                                           shuffle=shuffle,
                                           num_workers=4)

    def test_dataloader(self):
        ...
