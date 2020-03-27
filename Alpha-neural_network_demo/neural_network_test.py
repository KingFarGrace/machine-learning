#!/usr/bin/python3
# -*- encoding: utf-8 -*-

"""
@File : neural_network_test.py
@Time : 2020/03/27 20:15:55
@Author : KingFar 
@Version : 1.0
@Contact : 1136421682@qq.com
@WebSite : https://github.com/KingFarGrace
"""

# here put the import lib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import struct

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
测试集准确率
"""
def test_acc(paras):
    accuracy = [predict(test_set[0][index],
                parameters).argmax() == test_set[1][index]
                for index in range(test_num)]
    return accuracy.count(True) / len(accuracy)


"""
读取测试集数据
"""
def get_test_set(test_path):
    with open(test_path['test_img_path'], 'rb') as img_set:
        struct.unpack('>4i', img_set.read(16))
        imgs = np.fromfile(img_set, dtype=np.uint8).reshape(-1, 28 * 28)

    with open(test_path['test_lab_path'], 'rb') as lab_set:
        struct.unpack('>2i', lab_set.read(8))
        labs = np.fromfile(lab_set, dtype=np.uint8)

    return [imgs, labs]


if __name__ == '__main__':
    with open("./NN_model.pkl", "rb") as model_file:
        model_vars = pickle.load(model_file)
    util_func = (bypass, tanh, softmax)
    dimension = model_vars[0]
    parameters = model_vars[1]
    test_set_path = Path("./mnist_dataset")
    test_set_map = {'test_img_path': test_set_path / "t10k-images.idx3-ubyte",
                    'test_lab_path': test_set_path / "t10k-labels.idx1-ubyte"}
    test_set = get_test_set(test_set_map)
    test_num = 10000
    print("测试集准确率为：{}%".format(test_acc(parameters) * 100))
