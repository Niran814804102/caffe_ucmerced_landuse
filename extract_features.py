# -*- coding: utf-8 -*-

import sys

import numpy as np

caffe_root = '/home/zju/wlj/caffe-master/'  # 设置你caffe的安装目录
sys.path.insert(0, caffe_root + 'python')
import caffe  # 导入caffe

# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
print('load the structure of the model...')
model_def = '/home/zju/wlj/land_use_cnn/result/UCMerced_LandUse/deploy.prototxt'
print('load the weights of the model...')
model_weights = '/home/zju/wlj/land_use_cnn/result/UCMerced_LandUse/weights_finally.pretrained.caffemodel'

print('build the trained net...')
net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print('mean-subtracted values:', zip('BGR', mu),mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR


# caffe.io.load_image里有报错：_open() got an unexpected keyword argument 'as_grey'
# 因此用这个函数代替
def load_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    return image


def show_predict():
    # images = '/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/creat_lmdb.txt'
    images = '/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/train.txt'
    images = list(np.loadtxt(images, str, delimiter='\n'))
    features = []
    labels = []
    for image in images:
        true_label_num = int(image.split(' ')[1])
        image = load_image(image.split(' ')[0])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[0, ...] = transformed_image
        # 用fc7层的输入作为输出
        net.forward(start='conv1')
        feat = net.blobs['fc7'].data.copy()
        features.append(feat)
        labels.append(true_label_num)

    return features, labels


features, labels = show_predict()
np.save('/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/' + 'features_train.npy', features)
np.save('/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/' + 'labels_train.npy', labels)
