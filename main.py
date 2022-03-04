# -*- coding: utf-8 -*-
# date: 2022/3/1
# Project: DSFANet-Pytorch
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch.backends import cudnn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

from dataset import data_loader, ChDeDataset
from model.dsfanet import DSFANet, dsfa_loss
from utils.main import train_one_epoch, valid_one_epoch
from utils.utils import linear_sfa, kmeans, metric


class Train_and_Test_model():
    def __init__(self, data_name, device, seed, epochs, lr, reg, trn):
        '''
        :param data_name: 数据文件夹名称
        :param device: 训练设备
        :param epochs: 训练轮数
        :param lr: 学习率
        :param reg: 正则化参数
        :param trn: 训练数据的个数
        '''
        self.data_name = data_name
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.trn = trn

        # ------------------- 设置随机种子，使结果可以复现 --------------------
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        # -------------------- 加载数据 --------------------
        # self.train_dataloader = data_loader(data_name, True, trn)
        # self.valid_dataloader = data_loader(data_name, False, trn)  # 验证数据 和 测试 数据是同一波数据
        # self.test_dataloader = data_loader(data_name, False, trn)
        self.dataset = ChDeDataset(data_name, True, trn)
        self.x_train, self.y_trian, self.x_test, self.y_test = self.dataset[0]

        self.img_shape = self.dataset.img_shape  # (463, 241, 198)
        self.chg_map = self.dataset.chg_map
        # -------------------- 实例化 --------------------
        # 加载模型
        self.dsfanet = DSFANet(self.img_shape, train_num=trn, reg=reg)
        # 自定义初始化参数
        weights1 = torch.tensor(torch.normal(mean=0, std=1e-1, size=self.dsfanet.fc1.weight.shape))
        weights2 = torch.tensor(torch.normal(mean=0, std=1e-1, size=self.dsfanet.fc2.weight.shape))
        weights3 = torch.tensor(torch.normal(mean=0, std=1e-1, size=self.dsfanet.fc3.weight.shape))
        self.dsfanet.fc1.weight = torch.nn.Parameter(weights1)
        self.dsfanet.fc2.weight = torch.nn.Parameter(weights2)
        self.dsfanet.fc3.weight = torch.nn.Parameter(weights3)
        # 加载损失函数
        self.criterion = dsfa_loss()
        # 加载优化器
        self.optimizer = optim.SGD(self.dsfanet.parameters(), lr=lr)


    def train_model(self):
        best_acc_chg = 0
        self.dsfanet.train()
        for epoch in range(self.epochs):
            loss, V, fc_x, fc_y, B = train_one_epoch(self.dsfanet, self.x_train, self.y_trian, self.device, self.criterion, self.optimizer)

            mean_loss = loss / 6.
            print('epoch: {:03d} train_loss: {:4.4f}'.format(epoch+1, mean_loss))

            # 边 tain 边 valid
            total_loss, V, fc_x, fc_y, B = self.valid_model(self.dsfanet, self.x_test, self.y_test, self.device, self.criterion)
            imm, magnitude, differ_map = linear_sfa(fc_x.detach().numpy(), fc_y.detach().numpy(), V.detach().numpy(), shape=self.img_shape)
            magnitude = np.reshape(magnitude, self.img_shape[0:-1])
            change_map = np.reshape(kmeans(np.reshape(magnitude, [-1])), self.img_shape[0:-1])

            # acc_un, acc_chg, acc_all2, acc_tp = metric(1 - change_map, self.chg_map)
            acc_un, acc_chg, acc_all3, acc_tp = metric(change_map, self.chg_map)

            if acc_chg>best_acc_chg:
                # 根据验证集的表现保存最好的模型
                best_acc_chg = acc_chg
                best_model = self.dsfanet
                best_epoch = epoch+1

        total_loss, V, fc_x, fc_y, B = self.valid_model(best_model, self.x_test, self.y_test, self.device,self.criterion)
        imm, magnitude, differ_map = linear_sfa(fc_x.detach().numpy(), fc_y.detach().numpy(), V.detach().numpy(),shape=self.img_shape)
        magnitude = np.reshape(magnitude, self.img_shape[0:-1])
        change_map = np.reshape(kmeans(np.reshape(magnitude, [-1])), self.img_shape[0:-1])

        # acc_un, acc_chg, acc_all2, acc_tp = metric(1 - change_map, self.chg_map)
        acc_un, acc_chg, acc_all3, acc_tp = metric(change_map, self.chg_map)
        print('')
        print('Best Epoch: ',best_epoch)
        print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
        print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
        print('The True Positive ratio is:       %.4f' % (acc_tp))
        print('Accuracy of all testing sets is : %.4f' % (acc_all3))
        plt.imsave('results.png', change_map, cmap='gray')

    def valid_model(self, dsfanet, x_test, y_test, device, criterion):
        total_loss, V, fc_x, fc_y, B = valid_one_epoch(dsfanet, x_test, y_test, device, criterion)
        return total_loss, V, fc_x, fc_y, B

    def test_model(self):
        pass


if __name__ == '__main__':
    train_and_test = Train_and_Test_model(data_name='river',
                                          device='cpu',
                                          seed=0,
                                          epochs=100,
                                          lr=5*1e-5,
                                          reg=1e-4,
                                          trn=2000)
    train_and_test.train_model()