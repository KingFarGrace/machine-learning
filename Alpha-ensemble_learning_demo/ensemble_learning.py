#!/usr/bin/python3
# -*- encoding: utf-8 -*-

"""
@File : ensemble_learning.py
@Time : 2020/04/25 23:38:13
@Author : KingFar 
@Version : 1.0
@Contact : 1136421682@qq.com
@WebSite : https://github.com/KingFarGrace
"""

# here put the import lib
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    # 乳腺癌数据集存入文件
    # dataset = load_breast_cancer()
    # with open(r"dataset.pkl", "wb") as f:
    #     pickle.dump(dataset, f)
    # 读入数据集，按4: 1划分为训练集和测试集
    with open(r"dataset.pkl", "rb") as f:
        dataset = pickle.load(f)       
    datas, labels = dataset.data, dataset.target
    train_data, test_data, train_label, test_label = \
        train_test_split(datas, labels, test_size=0.2, random_state=0)
    # 将数据集转化为gbm中的数据格式
    gbm_train = lgb.Dataset(train_data, train_label)
    gbm_eval = lgb.Dataset(test_data, test_label, reference=gbm_train)

    # 训练轮次：50（本来设定成了100但是但是30多轮的时候就停下来了，所以砍一半）
    train_round = 50
    # 十轮训练没有准确率质变就提前停止
    early_stop_round = 10
    # 弱学习器参数，孩子用了都说好
    params = {
        'boosting_type': 'gbdt',    # 树的提升类型：gbdt
        'objective': 'binary',      # 目标函数：二分类任务
        'metric': {"l2", "auc"},    # 评估函数：方差评估，auc函数评估
        'num_leaves': 31,           # 最大叶节点数：缺省值
        'learning_rate': 0.1,       # 学习速率：缺省值
        'feature_fraction': 0.8,    # 特征选择率：4/5
        'bagging_fraction': 0.8,    # 建树采样比率：4/5
        'bagging_freq': 5,          # 每次执行bagging前的迭代轮次：5
        'verbose': 1                # I/O参数：显示所有信息
    }
    # 训练，保存预测值
    results = {}
    gbm = lgb.train(params, gbm_train,
                    num_boost_round=train_round,
                    valid_sets=(gbm_eval, gbm_train),
                    valid_names=("test", "train"),
                    early_stopping_rounds=early_stop_round,
                    evals_result=results)
    # 绘制准确率增长曲线以及特征重要程度直方图
    lgb.plot_metric(results)
    plt.show()
    lgb.plot_importance(gbm, importance_type="split")
    plt.show()
