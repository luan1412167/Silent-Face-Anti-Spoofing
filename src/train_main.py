# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm

import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet, MultiFTNetReload
from src.data_io.dataset_loader import get_data_loader
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class DepthFocalLoss(nn.Module):
    def __init__(self, gamma = 1, eps = 1e-7):
        super(DepthFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        loss = self.ce(input, target)
        loss = (loss) ** self.gamma
        return loss.mean()
class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every_epoch = conf.save_every_epoch
        self.step = 0
        self.start_epoch = 0
        self.train_loader, self.test_loader = get_data_loader(self.conf)

    def train_model(self):
        self._init_model_param()
        self._train_stage()
    
    def _init_model_param(self):
        # self.cls_criterion = CrossEntropyLoss()
        # self.ft_criterion = MSELoss()
        self.cls_criterion = FocalLoss()
        self.ft_criterion = DepthFocalLoss()

        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        # new 
        # self.optimizer = optim.AdamW(self.model.parameters(), lr= self.conf.lr,
        #                         weight_decay=5e-4)

        # self.schedule_lr = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, self.conf.milestones, self.conf.gamma, - 1)
        # new
        self.schedule_lr = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, verbose=True)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)


    def _train_stage(self):
        self.model.train()
        running_loss = 0.
        running_acc = []
        running_loss_cls = 0.
        running_loss_ft = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            self.step = e
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.optimizer.param_groups[0]['lr'])

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]
                labels = target

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc.append(acc)
            
            self.board_loss_every = len(running_acc)

            loss_board = running_loss / self.board_loss_every
            self.writer.add_scalar(
                'Training/Loss', loss_board, e)
            acc_board = sum(running_acc) / self.board_loss_every
            
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(
                'Training/Learning_rate', lr, e)
            loss_cls_board = running_loss_cls / self.board_loss_every
            
            loss_ft_board = running_loss_ft / self.board_loss_every
            self.writer.add_scalar(
                'Training/Loss_ft', loss_ft_board, e)

            if e % self.save_every_epoch == 0:
                time_stamp = get_time()
                self._save_state(time_stamp, extra=self.conf.job_name)

            total_loss_cls, total_acc = self._val_batch_data(self.test_loader)
            self.writer.add_scalars(
                'Training/Acc', {'training acc' : acc_board,
                                 'validate acc' : total_acc}, e)
            self.writer.add_scalars(
                'Training/Loss_cls', {'training loss cls' : loss_cls_board,
                                      'validate loss cls' : total_loss_cls}, e)
            print("Training epoch {} => running_loss_cls = {}, running_acc = {}".format(e, loss_cls_board, acc_board))
            print("Evaluate epoch {} => total_loss_cls = {}, total_acc = {}".format(e, total_loss_cls, total_acc))
            running_loss = 0.
            running_acc = []
            running_loss_cls = 0.
            running_loss_ft = 0.
            self.schedule_lr.step(acc_board)

        time_stamp = get_time()
        self._save_state(time_stamp, extra=self.conf.job_name)
        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))
        # embeddings = torch.sigmoid(embeddings)
        # print("embeddings ", embeddings, labels)

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _val_batch_data(self, val_data):
        self.model.eval()
        total_loss_cls, total_acc = 0, []
        with torch.no_grad():
            for sample, ft_sample, target in tqdm(iter(val_data)):
                imgs = [sample, ft_sample]
                labels = target

                labels = labels.to(self.conf.device)
                embeddings = self.model.forward(imgs[0].to(self.conf.device))
                # embeddings = torch.sigmoid(embeddings)
                total_loss_cls += self.cls_criterion(embeddings, labels)
                total_acc.append(self._get_accuracy(embeddings, labels)[0])
        return total_loss_cls / len(total_acc), sum(total_acc) / len(total_acc)

    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size}

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(self.get_state_dict(self.model), save_path + '/' +
                   ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))

    def get_state_dict(self, model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

class TrainMainPretrain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every_epoch = conf.save_every_epoch
        self.step = 0
        self.start_epoch = 0
        self.train_loader, self.test_loader = get_data_loader(self.conf)

    def train_model(self, model_path=''):
        self._init_model_param_pretrain(model_path)
        self._train_stage()

    def _init_model_param_pretrain(self, model_path=''):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network_pretrain(model_path)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        # new 
        # self.optimizer = optim.AdamW(self.model.parameters(), lr= self.conf.lr,
        #                         weight_decay=5e-4)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)
        # new
        # self.schedule_lr = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, verbose=True)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        self.model.train()
        running_loss = 0.
        running_acc = []
        running_loss_cls = 0.
        running_loss_ft = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.optimizer.param_groups[0]['lr'])
            self.step = e

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]
                labels = target

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc.append(acc)
            
            self.board_loss_every = len(running_acc)

            loss_board = running_loss / self.board_loss_every
            self.writer.add_scalar(
                'Training/Loss', loss_board, e)
            acc_board = sum(running_acc) / self.board_loss_every
            
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(
                'Training/Learning_rate', lr, e)
            loss_cls_board = running_loss_cls / self.board_loss_every
            
            loss_ft_board = running_loss_ft / self.board_loss_every
            self.writer.add_scalar(
                'Training/Loss_ft', loss_ft_board, e)

            if e % self.save_every_epoch == 0:
                time_stamp = get_time()
                self._save_state(time_stamp, extra=self.conf.job_name)

            total_loss_cls, total_acc = self._val_batch_data(self.test_loader)
            self.writer.add_scalars(
                'Training/Acc', {'training acc' : acc_board,
                                 'validate acc' : total_acc}, e)
            self.writer.add_scalars(
                'Training/Loss_cls', {'training loss cls' : loss_cls_board,
                                      'validate loss cls' : total_loss_cls}, e)
            print("Training epoch {} => running_loss_cls = {}, running_acc = {}".format(e, loss_cls_board, acc_board.item()))
            print("Evaluate epoch {} => total_loss_cls = {}, total_acc = {}".format(e, total_loss_cls, total_acc.item()))
            running_loss = 0.
            running_acc = []
            running_loss_cls = 0.
            running_loss_ft = 0.
            self.schedule_lr.step()

        time_stamp = get_time()
        self._save_state(time_stamp, extra=self.conf.job_name)
        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))
        # embeddings = torch.sigmoid(embeddings)

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _val_batch_data(self, val_data):
        self.model.eval()
        total_loss_cls, total_acc = 0, []
        with torch.no_grad():
            for sample, ft_sample, target in tqdm(iter(val_data)):
                imgs = [sample, ft_sample]
                labels = target
                # print(labels)
                labels = labels.to(self.conf.device)
                embeddings = self.model.forward(imgs[0].to(self.conf.device))
                # embeddings = torch.sigmoid(embeddings)
                total_loss_cls += self.cls_criterion(embeddings, labels)
                total_acc.append(self._get_accuracy(embeddings, labels)[0])
        return total_loss_cls / len(total_acc) , sum(total_acc) / len(total_acc)

    def _define_network_pretrain(self, model_path=""):
        model = MultiFTNetReload(model_path).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(self.get_state_dict(self.model), save_path + '/' +
                   ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))

    def get_state_dict(self, model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict
