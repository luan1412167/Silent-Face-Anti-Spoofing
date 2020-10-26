# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm

from datetime import datetime
import os
import torch
import numpy as np

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale

def parse_model_name_new_format(model_name):
    info = model_name.split('_')[0:-2]
    h_input, w_input = info[-1].split('x')
    model_type = "MiniFASNetV1SE"

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[3])
    return int(h_input), int(w_input), model_type, scale

def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


"""
Use in PyTorch.
"""

def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc


class BinaryClassificationMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0

    def update(self, pred, target):
        pred = torch.tensor(pred)
        target = torch.tensor(target)
        self.tp = pred.mul(target).sum(0).float()
        self.tn = (1 - pred).mul(1 - target).sum(0).float()
        self.fp = pred.mul(1 - target).sum(0).float()
        self.fn = (1 - pred).mul(target).sum(0).float()
        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
        self.avg_pre = np.nanmean(self.pre)
        self.avg_rec = np.nanmean(self.rec)
        self.avg_f1 = np.nanmean(self.f1)
        # print("tp, tn, fp, fn: ", self.tp.item(), self.tn.item(), self.fp.item(), self.fn.item())