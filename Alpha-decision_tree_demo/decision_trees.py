#!/usr/bin/python3
# -*- encoding: utf-8 -*-

"""
@File : decision_trees.py
@Time : 2020/03/09 23:56:06
@Author : KingFar
@Version : 1.0
@Contact : 1136421682@qq.com
@WebSite : https://github.com/KingFarGrace
"""

# here put the import lib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


"""
功能概述：数据集标准化函数，使用整型数据标记非数值类数据

功能具体描述：   从第一行开始，将数据集中同一列下不同的非数值类数据依次标记为0，1，2...
                相同的元素则归并到同一标记下且用标记覆盖原值，将产生的<原值:标记值>对加入字典中，每一列都重复上述过程。
                标记值会因原数据集排列不同而不同。

参数
--------------------
_data：外部数据集（列表），不包含表头。

返回值
--------------------
_data：被标记覆盖的原数据集，不包含表头；
attributes：数据集的 < 原值 : 标记值 > 字典。

参看
--------------------
无

示例
--------------------
    >>> data = [["A", "E", "F"], ["A", "C", "G"], ["B", "D", "F"]]
    >>> data, data_dict = data_standardizing(data)
    >>> print(data + '\n' + data_dict)
    [[0, 0, 1], [0, 1, 2], [0, 1, 0]]
    {'A': 0, 'B': 1, 'E': 0, 'C': 1, 'D': 2, 'F': 0, 'G': 1}    
"""


def data_standardizing(_data):
    attributes = {}
    #先列后行方式遍历数据集
    for i in range(len(_data[0])):
        count = 0
        for j in range(len(_data)):
            #若为新标签，则加入字典
            if _data[j][i] not in attributes:
                attributes[_data[j][i]] = count
                count += 1
            #用标签值覆盖原数据
            _data[j][i] = attributes[_data[j][i]]

    return _data, attributes


"""
功能概述：进行n次分类试验。

功能具体描述：对训练/测试集进行n次随机划分，每次划分都进行一次训练测试，并输出测试准确率。

参数
--------------------
_data：训练/测试数据集；
_label：标签集；
n：重复n次试验。

返回值
--------------------
sum_score / n：n次试验平均准确率

参看
--------------------
无

示例
--------------------
无

"""


def repeat_test(_data, _label, n):
    sum_score = 0
    for i in range(n):
        #每次从17个样本中随机选取10个作为训练样本，其余7个为测试样本
        train_data, test_data, train_label, test_label = \
            train_test_split(_data, _label, test_size=7 / 17)
        print("第{}次试验随机选取数据集：\n{}".format(i + 1, train_data))
        #以使用CART算法和最优解分类器的方式训练一颗随机取样最大深度为4的决策树
        classifier = DecisionTreeClassifier(criterion="gini", splitter="best",
                                            max_depth=4, random_state=None)
        classifier.fit(train_data, train_label)
        result = classifier.predict(test_data)
        acr_score = accuracy_score(result, test_label)
        sum_score += acr_score
        print("第{}次试验分类准确率：{}".format(i + 1, acr_score))
    return sum_score / n


if __name__ == '__main__':
    #读取西瓜数据集
    data_set = pd.read_csv("training_test_set.csv",
                           header=0, index_col=0, encoding="utf-8")
    #分离数据集和标签集
    data, label = data_set.iloc[:, 0: 6].values, data_set.iloc[:, 6].values
    #标签集序列化
    label = LabelEncoder().fit_transform(label)
    #数据集标准化
    data, data_dict = data_standardizing(data)
    #Start training!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("数据表如下：\n{}".format(data_dict))
    avg_score = repeat_test(data, label, 5)
    print("平均分类准确率为：{}".format(avg_score))
