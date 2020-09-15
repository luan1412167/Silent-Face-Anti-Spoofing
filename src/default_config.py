# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training
"""

import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel


def get_default_config():
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 1e-3
    # [9, 13, 15]
<<<<<<< HEAD
    conf.milestones = [100, 350, 500, 700]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 1000
    conf.momentum = 0.9
    conf.batch_size = 5
=======
    conf.milestones = [50, 150, 220]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 250
    conf.momentum = 0.9
    conf.batch_size = 64
>>>>>>> 5acd44edad9c058245ca5d650093503dcce65df4

    # model
    conf.num_classes = 2
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.train_root_path = './datasets/RGB_Images'

    # save file path
    conf.snapshot_dir_path = './saved_logs/snapshot'

    # log path
    conf.log_path = './saved_logs/jobs'
    # tensorboard
    conf.board_loss_every = 1000
    # save model/iter
<<<<<<< HEAD
    conf.save_every = 50
=======
    conf.save_every_epoch = 3
>>>>>>> 5acd44edad9c058245ca5d650093503dcce65df4

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
