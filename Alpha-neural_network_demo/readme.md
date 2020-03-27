# 神经网络
+ 此文件用于构建一个包含一个隐层的简单神经网络
+ 通过`numpy`手动实现梯度下降算法以及其他神经网络必需函数，不使用`TensorFlow`和`PyTorch`等包含神经网络框架的第三方库
+ 文件列表：
    > mnist_dataset：存放手写数字图片数据集的文件；
      neural_network_training.py：神经网络训练源码；
      neural_network_test.py：神经网络测试源码；
      NN_model.pkl：神经网络模型数据。
+ mnist_dataset：MNIST手写数字图片集，包含：
    1. t10k-images.idx3-ubyte：测试图片集
    2. t10k-labels.idx1-ubyte：测试标签集
    3. train-images.idx3-ubyte：训练图片集
    4. train-labels.idx1-ubyte：训练标签集
+ neural_network_training.py：单隐层神经网络训练
    1. 学习率：10^(-1.34)
    2. 每批训练100个图片，一个纪元训练500批，共训练100个纪元
    3. 最终验证集准确率约为92.07%，测试集准确率约为92.34%
    4. 100个训练纪元内未出现过拟合情况
    5. 输入层-隐层不使用激活函数处理，隐层-输出层激活函数采用`tanh`，输出层损失函数采用`softmax`
+ neural_network_test.py：手写图片识别准确率，从NN_model中读取模型和参数
+ NN_model.pkl：存放模型以及训练过的参数