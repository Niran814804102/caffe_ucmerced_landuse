# Land_Use_CNN
## 1. CNN
1. data/data_prepare.py:数据准备
2. train.py:微调CNN网络
3. predict.py:使用训练好的CNN模型对图片经行分类：CPU3s预测一张，GPU0.03s预测一张
4. t_sne.py:对CNN提取到的特征降维可视化

## 2. SVM
1. extract_features.py：提取CNN中fc7层图片特征
2. train_svm.py: 利用f7提取到的特征训练SVM，精度0.935
3. svm_predict.py:利用CNN和SVM对图片分类，GPU0.3s预测一张
