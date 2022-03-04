# -*- coding: utf-8 -*-
# date: 2022/3/1
# Project: DSFANet-Pytorch
# File Name: dataset.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com


import torch
from torch.utils.data import Dataset, DataLoader

from utils.utils import load_data, getTrainSamples


class ChDeDataset(Dataset):
    def __init__(self, data_name, train, numbers):
        '''
        :param data_path: 要加载的数据的名称
        :param train: 生成训练True 或 测试数据集False
        :param numbers: 划分数据后要参与训练的数据的 个数
        '''
        super(ChDeDataset, self).__init__()
        self.data_name = data_name
        self.train = train
        self.numbers = numbers

        # ----------- 加载数据 ---------------
        self.im1, self.im2, self.chg_ref, self.cva_ind, self.img_shape, self.chg_map = load_data(data_name)
        self.x_train, self.y_train = getTrainSamples(self.cva_ind, self.im1, self.im2, numbers)

    def __getitem__(self, index):
        # if self.train:
        #     x_train = torch.from_numpy(self.x_train.transpose(1,0)).type(torch.FloatTensor)
        #     y_trian = torch.from_numpy(self.y_train.transpose(1,0)).type(torch.FloatTensor)
        #     return x_train, y_trian
        # else:
        #     x_test = torch.from_numpy(self.im1.transpose(1,0)).type(torch.FloatTensor)
        #     y_test = torch.from_numpy(self.im2.transpose(1,0)).type(torch.FloatTensor)
        #     return x_test, y_test

        x_train = torch.from_numpy(self.x_train.transpose(1, 0)).type(torch.FloatTensor)
        y_trian = torch.from_numpy(self.y_train.transpose(1,0)).type(torch.FloatTensor)
        x_test = torch.from_numpy(self.im1.transpose(1, 0)).type(torch.FloatTensor)
        y_test = torch.from_numpy(self.im2.transpose(1,0)).type(torch.FloatTensor)
        return x_train, y_trian, x_test, y_test

    def __len__(self):
        if self.train:
            return self.x_train.shape[0]
        else:
            return self.im1.shape[0]

    def data_info(self):
        print('*'*10, 'Data Info', '*'*20)
        print('x_train shape:{}\t y_train shape:{}'.format(self.x_train.shape, self.y_train.shape))
        print('x_test shape:{}\t y_test shape:{}'.format(self.im1.shape, self.im2.shape))
        print('*' * 10, 'Data Info', '*' * 20)


def data_loader(data_name, train, numbers):
    '''获取dataloader'''
    dataset = ChDeDataset(data_name, train, numbers)
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader

if __name__ == '__main__':
    dataloader = data_loader('river', True, 2000)
    print(len(dataloader))
    x,y = iter(dataloader).next()
    print(x.shape, y.shape)