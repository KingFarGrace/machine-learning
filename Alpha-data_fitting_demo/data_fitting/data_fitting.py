#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import re


'''
从文件中滤出横纵坐标数据
'''
def file_data_filter(file_path):
    axis_set = []
    axis_set_x = []
    axis_set_y = []
    file = open(file_path, mode='r')
    str = file.readlines()
    for i in range(len(str)):
        # 匹配字符串中所有数字的正则表达式
        axis = re.findall(r"\d+\.?\d*", str[i])
        axis_set += [float(n) for n in axis]
    print(axis_set, '\n')
    for i in range(len(axis_set)):
        if i % 2 == 0:
            axis_set_x.append(axis_set[i])
        else:
            axis_set_y.append(axis_set[i])
    file.close()
    return axis_set_x, axis_set_y


'''
拟合数据点并绘制曲线
'''
def get_fitting_data(axis_set_x, axis_set_y, deg):
    # 生成多项式
    index = np.polyfit(axis_set_x, axis_set_y, deg)
    formula = np.poly1d(index)
    print(deg, "次多项式参数为：", index, "（从高次项开始依次向后排列）", '\n')
    # 生成绘图坐标
    x_f = np.arange(0, np.amax(axis_set_x), 1)
    y_f = [formula(n) for n in x_f]
    # 绘图
    plt.scatter(axis_set_x, axis_set_y)
    plt.plot(x_f, y_f, '-r')
    plt.show()
    return index


if __name__ == '__main__':
    # 滤出一次数据和二次数据
    line_file_path = r"D:\MyPython\linear_data.txt"
    line_axis_set_x, line_axis_set_y = file_data_filter(line_file_path)

    quad_file_path = r"D:\MyPython\quadratic_data.txt"
    quad_axis_set_x, quad_axis_set_y = file_data_filter(quad_file_path)

    # 拟合两组数据并将拟合参数写入文件
    line_index = get_fitting_data(line_axis_set_x, line_axis_set_y, 1)
    quad_index = get_fitting_data(quad_axis_set_x, quad_axis_set_y, 2)
    file = open(r"D:\MyPython\index.txt", mode='w')
    file.write("一次多项式参数为：" + str(line_index) + "（从高次项开始依次向后排列）" + '\n')
    file.write("二次多项式参数为：" + str(quad_index) + "（从高次项开始依次向后排列）" + '\n')
    file.close()
