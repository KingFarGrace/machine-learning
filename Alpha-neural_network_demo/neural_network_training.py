#!/usr/bin/python3
# -*- encoding: utf-8 -*-

"""
@File : neural_network_training.py
@Time : 2020/03/25 22:43:58
@Author : KingFar 
@Version : 1.0
@Contact : 1136421682@qq.com
@WebSite : https://github.com/KingFarGrace
"""

# here put the import lib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import copy
import tqdm
import pickle

"""
输入层到隐层层的激励函数
"""
def bypass(x):
    return x


"""
隐层到输出层的激励函数
"""
def tanh(x):
    return np.tanh(x)


"""
输出层损失函数
"""
def softmax(x):
    exp = np.exp(x - x.max())   # minus the max value to avoid exp-explosion
    return exp / exp.sum()


"""
输入层到隐层层的激励函数导数
"""
def d_bypass(x):
    return 1


"""
隐层到输出层的激励函数导数
"""
def d_tanh(x):
    return 1 / np.cosh(x) ** 2


"""
输出层损失函数导数
"""
def d_softmax(x):
    data = softmax(x)
    return np.diag(data) - np.outer(data, data)


"""
求预计输出矩阵和实际输出矩阵的平方差
"""
def sqr_loss(img, lab, paras):
    x_predict = predict(img, paras)
    x_real = np.identity(dimension[-1])
    diff = x_real - x_predict
    return np.dot(diff, diff)


"""
初始化参数b
"""
def init_para_b(layer):
    para = distribution[layer]['b']
    return np.random.rand(dimension[layer]) * (para[1] - para[0]) + para[0]


"""
初始化参数w
"""
def init_para_w(layer):
    para = distribution[layer]['w']
    return np.random.rand(dimension[layer - 1], dimension[layer]) * (para[1] - para[0]) + para[0]


"""
随机初始化参数
"""
def init_paras():
    paras = []
    for i in range(len(distribution)):
        layer_paras = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_paras['b'] = init_para_b(i)
                continue
            if j == 'w':
                layer_paras['w'] = init_para_w(i)
                continue
        paras.append(layer_paras)
    return paras


"""
读取训练集并按5:1划分为训练集和验证集
"""
def get_train_valid_set(tv_path):
    with open(tv_path['train_data_path'], 'rb') as data_set:
        struct.unpack('>4i', data_set.read(16))
        temp_data = np.fromfile(data_set, dtype=np.uint8).reshape(-1, 28 * 28)
        train_data = temp_data[: train_num]
        valid_data = temp_data[train_num:]

    with open(tv_path['train_label_path'], 'rb') as lab_set:
        struct.unpack('>2i', lab_set.read(8))
        temp_lab = np.fromfile(lab_set, dtype=np.uint8)
        train_lab = temp_lab[: train_num]
        valid_lab = temp_lab[train_num:]

    return [[train_data, train_lab], [valid_data, valid_lab]]


"""
输入手写数字图像，经过神经网络，输出判断值（0-9）
"""
def predict(img, paras):
    data_in = img
    data_out = util_func[0](data_in)
    for layer in range(1, len(dimension)):
        data_in = np.dot(data_out, parameters[layer]['w']) + parameters[layer]['b']
        data_out = util_func[layer](data_in)
    return data_out


"""
梯度下降算法，按着西瓜书上的公式一点点调整的，原理有点模糊，所以不写注释了。
"""
def grad_decrease(img, lab, paras):
    data_in_list = [img]
    data_out_list = [util_func[0](data_in_list[0])]
    for layer in range(1, len(dimension)):
        data_in = np.dot(data_out_list[layer - 1], parameters[layer]['w']) + parameters[layer]['b']
        data_out = util_func[layer](data_in)
        data_in_list.append(data_in)
        data_out_list.append(data_out)
    diff = -2 * (np.identity(dimension[-1])[lab] - data_out_list[-1])
    grad = [None] * len(dimension)
    for layer in range(len(dimension) - 1, 0, -1):
        if multi_type[util_func[layer]] == "multiply":
            diff *= util_diff[util_func[layer]](data_in_list[layer])
        if multi_type[util_func[layer]] == "dot":
            diff = np.dot(util_diff[util_func[layer]](data_in_list[layer]), diff)
        grad[layer] = {}
        grad[layer]['b'] = diff
        grad[layer]['w'] = np.outer(data_out_list[layer - 1], diff)
        diff = np.dot(parameters[layer]['w'], diff)
    return grad


"""
计算训练集偏差值
"""
def train_loss(paras):
    loss = 0
    for index in range(train_num):
        loss += sqr_loss(data_lab_set[0][0][index], data_lab_set[0][1][index], parameters)
    return loss / (train_num / 10000)


"""
计算训练集准确率
"""
def train_acc(paras):
    accuracy = [predict(data_lab_set[0][0][index],
                parameters).argmax() == data_lab_set[0][1][index]
                for index in range(train_num)]
    return accuracy.count(True) / len(accuracy)


"""
计算验证集偏差值
"""
def valid_loss(paras):
    loss = 0
    for index in range(valid_num):
        loss += sqr_loss(data_lab_set[1][0][index], data_lab_set[1][1][index], parameters)
    return loss / (valid_num / 10000)


"""
计算验证集准确率
"""
def valid_acc(paras):
    accuracy = [predict(data_lab_set[1][0][index],
                parameters).argmax() == data_lab_set[1][1][index]
                for index in range(train_num)]
    return accuracy.count(True) / len(accuracy)


def grad_add(grad1, grad2):
    for layer in range(1, len(grad1)):
        for k in grad1[layer].keys():
            grad1[layer][k] += grad2[layer][k]
    return grad1


def grad_divide(grad, denominator):
    for layer in range(1, len(grad)):
        for k in grad[layer].keys():
            grad[layer][k] /= denominator
    return grad


"""
按组训练神经网络，每组训练100个图片，并做梯度下降
"""
def group_training(current_group, paras):
    grad_sum = grad_decrease(data_lab_set[0][0][current_group * group_size + 0],
                             data_lab_set[0][1][current_group * group_size + 0],
                             parameters)
    # print(grad_sum)
    for index in range(1, group_size):
        temp_grad = grad_decrease(data_lab_set[0][0][current_group * group_size + index],
                                  data_lab_set[0][1][current_group * group_size + index],
                                  parameters)
        grad_add(grad_sum, temp_grad)
    grad_divide(grad_sum, group_size)

    return grad_sum


"""
根据当前梯度和学习率调整参数，公式抄书来的
"""
def para_justify(paras, grad, learn_rate):
    temp_paras = copy.deepcopy(paras)
    for layer in range(1, len(temp_paras)):
        for k in temp_paras[layer].keys():
            temp_paras[layer][k] -= learn_rate * grad[layer][k]
    return temp_paras


"""
将神经网络模型存入文件中
"""
def save_model(*model_vars):
    with open("./NN_model.pkl", "wb") as model_file:
        pickle.dump(model_vars, model_file)


if __name__ == '__main__':
    dataset_path = Path("./mnist_dataset")
    path_map = {'train_data_path': dataset_path/"train-images.idx3-ubyte",
                'train_label_path': dataset_path/"train-labels.idx1-ubyte"}

    # 三层网络模型：输入层28*28个神经元，隐层100个，输出层10个
    dimension = (28 * 28, 100, 10)
    # 工具函数
    util_func = (bypass, tanh, softmax)
    # 工具导数，通过工具函数访问
    util_diff = {bypass: d_bypass, tanh: d_tanh, softmax: d_softmax}
    # mutiply表示函数可以直接做乘法（经过优化），dot表明只能做矩阵乘法（未经优化）。
    multi_type = {bypass: 'multiply', tanh: 'multiply', softmax: 'dot'}
    # 参数模型，w为什么在这个范围里我也不知道，问就是抄的经验参数
    distribution = [
        {},     # no parameters needed at the input-layer
        {'b': [0, 0], 'w': [-np.sqrt(6 / (dimension[0] + dimension[1])),
                            np.sqrt(6 / (dimension[0] + dimension[1]))]},
        {'b': [0, 0], 'w': [-np.sqrt(6 / (dimension[1] + dimension[2])),
                            np.sqrt(6 / (dimension[1] + dimension[2]))]}
    ]

    # 训练集:验证集 = 5:1
    train_num = 50000
    valid_num = 10000
    data_lab_set = get_train_valid_set(path_map)
    parameters = init_paras()
    # print(parameters)

    # 一组训练一百个图像，一个纪元训练500组，训练100个纪元
    group_size = 100
    epoch = 100
    # 调这个学习率调了一晚上。。。
    learn_rate = 10 ** -1.34
    # print(learn_rate)
    current_epoch = 0
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch_index in tqdm.tqdm_notebook(range(epoch)):
        for i in range(train_num // group_size):
            print("epoch{}:{}/{}".format(epoch_index + 1, i + 1, train_num // group_size))
            grad_t = group_training(i, parameters)
            parameters = para_justify(parameters, grad_t, learn_rate)
        current_epoch += 1
        train_loss_list.append((train_loss(parameters)))
        train_acc_list.append((train_acc(parameters)))
        valid_loss_list.append((train_loss(parameters)))
        valid_acc_list.append((train_acc(parameters)))
    print("t_loss:{}\nt_acc:{}\nv_loss:{}\nv_acc:{}".
          format(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list))
    save_model(dimension, parameters)
