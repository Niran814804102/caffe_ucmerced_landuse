# -*- coding: utf-8 -*-

from time import time

import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

caffe_root = '/home/zju/wlj/caffe-master/'  # 设置你caffe的安装目录
X = np.load('/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/' + 'features_train.npy')
y = np.load('/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/' + 'labels_train.npy')
# y=y.reshape((-1,1))

# X.shape=(1680, 50, 4096) Y.shape=(1680)
# 1680
# X = X.reshape((1680, 4096))
# 第二个维度用max来去掉
X = X.max(axis=1)
print(X.shape, y.shape)
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Fitting the classifier to the training set")
t0 = time()
# C越大，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱
clf = SVC(C=10000, probability=True).fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test)

print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
# 准备模型输出目录
import os
model_dir = '/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/' + 'svm_model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
joblib.dump(clf, model_dir + 'svm.pkl')
