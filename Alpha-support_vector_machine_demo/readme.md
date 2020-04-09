# 支持向量机
本项目包含以下文件：
+ mnist_dataset：手写数字图片数据集
    1. t10k-images.idx3-ubyte：测试图片集
    2. t10k-labels.idx1-ubyte：测试标签集
    3. train-images.idx3-ubyte：训练图片集
    4. train-labels.idx1-ubyte：训练标签集
    
+ support_vector_machine.py：支持向量机源码
    1. 采用sklearn中的svm框架快速搭建（其实最开始写了一个手动搭建的模型，但是自己写的数学模型跑不动这么大的数据集，总而言之，sklearn真香！）
    2. 模型参数：
    ``` python
    SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    ```
    3. 最终预测结果：
    
    |              | precision | recall | f1-score | support |
    | :----------: | :-------: | :----: | :------: | :-----: |
    |      0       |   0.98    |  0.99  |   0.99   |   980   |
|      1       |   0.99    |  0.99  |   0.99   |  1135   |
    |      2       |   0.97    |  0.97  |   0.97   |  1032   |
    |      3       |   0.96    |  0.97  |   0.97   |  1010   |
    |      4       |   0.97    |  0.98  |   0.98   |   982   |
    |      5       |   0.98    |  0.98  |   0.98   |   892   |
    |      6       |   0.99    |  0.99  |   0.99   |   958   |
    |      7       |   0.98    |  0.96  |   0.97   |  1028   |
    |      8       |   0.97    |  0.97  |   0.97   |   974   |
    |      9       |   0.98    |  0.95  |   0.96   |  1009   |
    |   accuracy   |           |        |   0.98   |  10000  |
    |  macro avg   |   0.98    |  0.98  |   0.98   |  10000  |
| weighted avg |   0.98    |  0.98  |   0.98   |  10000  |
    
   