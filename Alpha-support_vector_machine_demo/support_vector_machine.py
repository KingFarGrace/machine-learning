#!/usr/bin/python3
# -*- encoding: utf-8 -*-

"""
@File : support_vector_machine.py
@Time : 2020/04/09 21:35:38
@Author : KingFar 
@Version : 1.0
@Contact : 1136421682@qq.com
@WebSite : https://github.com/KingFarGrace
"""

# here put the import lib
from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# 读取图像文件
def get_img(path):
    with open(path, 'rb') as imgf:
        struct.unpack('>4i', imgf.read(16))
        temp_img = np.fromfile(imgf, dtype=np.uint8).reshape(-1, 28 * 28)
    return temp_img

# 读取标签文件
def get_lab(path):
    with open(path, 'rb') as labf:
        struct.unpack('>2i', labf.read(8))
        temp_lab = np.fromfile(labf, dtype=np.uint8)
    return temp_lab


"""
功能概述：获得一个对当前分类任务最合适的svm模型

功能具体描述：通过网格搜索，五折交叉验证从参数列表中挑选最合适的参数组合

参数
--------------------
param: 所有可能的参数组合列表，每组参数（即列表的每项元素）以字典的形式存放；
grid: 经过5折交叉验证得网格搜索确定的参数模型组合；

返回值
--------------------
grid.best_estimator_: 最好的参数模型

参看
--------------------
此函数为测试阶段调用并获得最佳模型，在程序运行时不调用。

示例
--------------------
>>> svc_model = get_best_model()
>>> print(svc_model)
    SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False).fit(train_img_std, train_lab)

"""
def get_best_model():
    """
    参数列表：
    线性核-C值:[1, 5, 10, 50]
    高斯核-C值:[1, 5, 10, 50]-核函数参数gamma:[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    多项式核-C值:[1, 5, 10, 50]-核函数参数degree:[2, 3, 4], gamma:自动选择
    """
    param = [
        {'kernel': ['linear'], 'C': [1, 5, 10, 50]},
        {'kernel': ['rbf'], 'C': [1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]},
        {'kernel': ['poly'], 'C': [1, 5, 10, 50], 'degree':[2, 3, 4], 'gamma': ['auto']}
    ]
    # 参数：自动最优匹配的SVC，参数列表param，5折交叉验证
    grid = GridSearchCV(SVC(class_weight="balanced"), param, cv=5)
    grid.fit(train_img_std, train_lab)
    return grid.best_estimator_


if __name__ == '__main__':
    dataset_path = Path("Alpha-support_vector_machine_demo/mnist_dataset")
    path_map = {'train_img_path': dataset_path / "train-images.idx3-ubyte",
                'train_lab_path': dataset_path / "train-labels.idx1-ubyte",
                'test_img_path': dataset_path / "t10k-images.idx3-ubyte",
                'test_lab_path': dataset_path / "t10k-labels.idx1-ubyte"}
    train_img = get_img(path_map['train_img_path'])
    train_lab = get_lab(path_map['train_lab_path'])
    test_img = get_img(path_map['test_img_path'])
    test_lab = get_lab(path_map['test_lab_path'])

    # PCA特征提取，从28*28维图片中提取100维特征
    cpn = 100
    pca = PCA(n_components=cpn, svd_solver="randomized", whiten=True, random_state=None).fit(train_img)
    train_img_std = pca.transform(train_img)
    test_img_std = pca.transform(test_img)

    # svc_model = get_best_model()
    # print(svc_model)
    # 最佳模型
    svc_model = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False).fit(train_img_std, train_lab)
    p_lab = svc_model.predict(test_img_std)
    # 打印分类报表
    print(classification_report(test_lab, p_lab))
