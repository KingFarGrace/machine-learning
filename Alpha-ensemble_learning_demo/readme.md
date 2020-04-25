# 集成学习

此集成学习程序采用lightGBM工具包和sklearn库进行快速搭建，参数部分缺省。

+ dataset.pkl：存放乳腺癌数据集（load_breast_cancer()返回值：一个对象）。

+ ensemble_learning.py：集成学习源代码文件。

  + 弱学习器参数：
  ```python
      params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': {"l2", "auc"},
          'num_leaves': 31,
          'learning_rate': 0.1,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 5,
          'verbose': 1
      }
  ```

这次学习任务大概是对我这个民工电脑最友好的了，感谢lightGBM的神优化。本来是想要尝试一下比赛的数据集的，但是似乎缺少其他方面的知识，只好先作罢，待日后卷土重来（

学习准确率高达99.8%，怪不得竞赛高分大多都是集成学习，这准确率SVM都比不了，而且跑的还快。