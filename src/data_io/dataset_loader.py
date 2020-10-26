# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans
import torch


def get_data_loader(conf):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.2),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        # trans.Normalize(mean, std)
    ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    trainset = DatasetFolderFT(root_path + "/train" , train_transform,
                               None, conf.ft_width, conf.ft_height)
    testset = DatasetFolderFT(root_path + "/val" , train_transform,
                               None, conf.ft_width, conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8)

    test_loader = DataLoader(
        testset,
        batch_size=conf.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8)
    return train_loader, test_loader
