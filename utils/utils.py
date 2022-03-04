# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio
from matplotlib import image
from scipy.cluster.vq import kmeans as km


def metric(img=None, chg_ref=None):

    chg_ref = np.array(chg_ref, dtype=np.float32)
    chg_ref = chg_ref / np.max(chg_ref)

    img = np.reshape(img, [-1])
    chg_ref = np.reshape(chg_ref, [-1])

    loc1 = np.where(chg_ref == 1)[0]
    num1 = np.sum(img[loc1] == 1)
    acc_chg = np.divide(float(num1), float(np.shape(loc1)[0]))

    loc2 = np.where(chg_ref == 0)[0]
    num2 = np.sum(img[loc2] == 0)
    acc_un = np.divide(float(num2), float(np.shape(loc2)[0]))

    acc_all = np.divide(float(num1 + num2), float(np.shape(loc1)[0] + np.shape(loc2)[0]))

    loc3 = np.where(img == 1)[0]
    num3 = np.sum(chg_ref[loc3] == 1)
    acc_tp = np.divide(float(num3), float(np.shape(loc3)[0]))

    # print('')
    # print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
    # print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
    # print('The True Positive ratio is:       %.4f' % (acc_tp))
    # print('Accuracy of all testing sets is : %.4f' % (acc_all))

    return acc_un, acc_chg, acc_all, acc_tp


def getTrainSamples(index, im1, im2, number=4000):
    # 获取参与训练的像素点所对应的索引
    loc = np.where(index != 1)[0]
    # np.random.permutation()：随机排列序列
    # 对数据进行类似shuffle的操作
    perm = np.random.permutation(np.shape(loc)[0])
    # 根据传入的number的值选取 number个参与训练的像素点
    ind = loc[perm[0:number]]

    return im1[ind, :], im2[ind, :]


def normlize(data):
    meanv = np.mean(data, axis=0)
    stdv = np.std(data, axis=0)

    delta = data - meanv
    data = delta / stdv

    return data


def linear_sfa(fcx, fcy, vp, shape):

    delta = np.matmul(fcx, vp) - np.matmul(fcy, vp)

    #delta = delta / np.std(delta, axis=0)

    delta = delta**2

    differ_map = delta#normlize(delta)

    magnitude = np.sum(delta, axis=1)

    vv = magnitude / np.max(magnitude)

    im = np.reshape(kmeans(vv), shape[0:-1])

    return im, magnitude, differ_map


def load_data(data_name):
    '''加载数据集
    :return im1: 第一个时相数据
            im2: 第二个时相数据
            chg_ref:
    '''
    img1_path = os.path.join('data', data_name, 'img_t1.mat')  # 第一个时相t1的数据
    img2_path = os.path.join('data', data_name, 'img_t2.mat')  # 第二个时相t2的数据
    change_path = os.path.join('data', data_name, 'chg_ref.bmp')
    ind_path = os.path.join('data', data_name, 'cva_ref.mat')  # load cva pre-detection result

    mat1 = sio.loadmat(img1_path)
    mat2 = sio.loadmat(img2_path)
    ind = sio.loadmat(ind_path)

    img1 = mat1['im']
    img2 = mat2['im']
    chg_map = image.imread(change_path)  # 用于记录哪些位置是背景0，哪些是物体1
    cva_ind = ind['cva_ref']  # 用于划分哪些数据需要进行训练

    im1, im2, chg_ref, cva_ind = reshape_normlize_data(img1, img2, chg_map, cva_ind)

    return im1, im2, chg_ref, cva_ind, img1.shape, chg_map


def reshape_normlize_data(img1, img2, chg_map, cva_ind):
    img_shape = np.shape(img1)  # [h, w, c]
    # im1 和 im2可以理解为两个时相的数据
    # im1相当于要参与训练的数据，im2相当于要训练数据对应的 标签
    im1 = np.reshape(img1, newshape=[-1, img_shape[-1]])  # [h*w, c]
    im2 = np.reshape(img2, newshape=[-1, img_shape[-1]])

    im1 = normlize(im1)  # 将数据标准化
    im2 = normlize(im2)

    chg_ref = np.reshape(chg_map, newshape=[-1])  # 将记录点变为一维的
    cva_ind = np.reshape(cva_ind, newshape=[-1])  # 变为一维的

    return im1, im2, chg_ref, cva_ind


def kmeans(data):
    shape = np.shape(data)
    # print((data))
    ctr, _ = km(data, 2)

    for k1 in range(shape[0]):
        if abs(ctr[0] - data[k1]) >= abs(ctr[1] - data[k1]):
            data[k1] = 0
        else:
            data[k1] = 1
    return data
