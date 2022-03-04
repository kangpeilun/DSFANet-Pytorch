# -*- coding: utf-8 -*-
# date: 2022/3/2
# Project: DSFANet-Pytorch
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
import numpy as np


def train_one_epoch(dsfanet, x_train, y_train, device, criterion, optimizer):
    total_loss = []
    # x_train, y_train = iter(train_dataloader).next()
    # x_train:[1, 198, 2000]  y_train:[1, 198, 2000]
    x_train, y_train = x_train.to(device), y_train.to(device)
    # ============ forward ==============
    sigma, V, fc_x, fc_y, B = dsfanet(x_train, y_train)
    loss = criterion(sigma)

    # total_loss.append(loss.item())

    # =========== backward =============
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, V, fc_x, fc_y, B


def valid_one_epoch(dsfanet, x_test, y_test, device, criterion):
    total_loss = []
    # x_train, y_train = iter(train_dataloader).next()
    # x_train:[1, 198, 2000]  y_train:[1, 198, 2000]
    x_test, y_test = x_test.to(device), y_test.to(device)
    # ============ forward ==============
    sigma, V, fc_x, fc_y, B = dsfanet(x_test, y_test)
    loss = criterion(sigma)

    total_loss.append(loss.item())

    return loss, V, fc_x, fc_y, B