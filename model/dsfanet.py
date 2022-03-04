# -*- coding: utf-8 -*-
# date: 2022/3/1
# Project: DSFANet-Pytorch
# File Name: dsfanet.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch import nn


class dsfa_loss(nn.Module):
    def __init__(self):
        super(dsfa_loss, self).__init__()

    def forward(self, sigma):
        return torch.trace(torch.matmul(sigma.transpose(1, 0), sigma))



class DSFANet(nn.Module):
    def __init__(self, shape, train_num, reg=1e-4):
        '''
        :param shape: (463, 241, 198)
        :param reg: regularization parameter
        '''
        super(DSFANet, self).__init__()
        self.reg = reg
        self.train_num = train_num
        self.bands = shape[2]

        self.net_shape = [128, 128, 6]

        self.softplus = nn.Softplus()  # 激活函数
        self.fc1 = nn.Linear(self.bands, self.net_shape[0])
        self.fc2 = nn.Linear(self.net_shape[0], self.net_shape[1])
        self.fc3 = nn.Linear(self.net_shape[1], self.net_shape[2])

    def get_sigma(self, Differ, fc_x, fc_y):
        A = torch.matmul(Differ.transpose(1, 0), Differ)  # 矩阵乘法
        A = A / self.train_num

        sigmaX = torch.matmul(fc_x.transpose(1, 0), fc_x)
        sigmaY = torch.matmul(fc_y.transpose(1, 0), fc_y)
        sigmaX = sigmaX/self.train_num + self.reg*torch.eye(self.net_shape[-1])
        sigmaY = sigmaY/self.train_num + self.reg*torch.eye(self.net_shape[-1])

        B = (sigmaX + sigmaY) / 2

        D_B, V_B = torch.eig(B, eigenvectors=True)  # 特征值，特征向量
        idx = torch.where(D_B > 1e-12)[0]
        D_B = torch.tensor([D[0] for D in D_B.detach()])
        temp_D = torch.zeros(size=(idx.shape[0],))
        temp_V = torch.zeros(size=(V_B.shape[0],idx.shape[0]))
        for i, index in enumerate(idx):  # i从0起始用于想temp中存入数据，index根据idx生成用于筛选数据
            temp_D[i] = D_B[index]
            temp_V[i, :] = V_B[index, :]
        D_B = temp_D
        V_B = temp_V

        B_inv = torch.matmul(torch.matmul(V_B, torch.diag(torch.reciprocal(D_B))), V_B.transpose(1, 0))
        sigma = torch.matmul(B_inv, A)

        return sigma, B

    def forward(self, x, y):
        # batch_size=1
        # x: [batch_size, bands, numbers]  t1
        # y: [batch_size, bands, numbers]  t2
        # x = x.permute(0, 2, 1)
        # y = y.permute(0, 2, 1)  # [numbers, bands]
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)
        x = self.softplus(self.fc1(x))
        y = self.softplus(self.fc1(y))

        x = self.softplus(self.fc2(x))
        y = self.softplus(self.fc2(y))

        fc3x = self.softplus(self.fc3(x)).squeeze()  # [numbers, 6]
        fc3y = self.softplus(self.fc3(y)).squeeze()

        m = fc3x.size(1)
        fc_x = fc3x - torch.divide(torch.tensor(1.), torch.tensor(m)) * torch.matmul(fc3x, torch.ones([m, m]))
        fc_y = fc3y - torch.divide(torch.tensor(1.), torch.tensor(m)) * torch.matmul(fc3y, torch.ones([m, m]))

        Differ = fc_x - fc_y
        # Differ =  fc_y - fc_x
        sigma, B = self.get_sigma(Differ, fc_x, fc_y)
        D, V = torch.eig(sigma, eigenvectors=True)

        return sigma, V, fc_x, fc_y, B


if __name__ == '__main__':
    model = DSFANet((463, 241, 198), 2000)
    print(model)
    print(model.fc1.weight.shape)
    print(model.fc2.weight.shape)
    print(model.fc3.weight.shape)