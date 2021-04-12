# 1. 准备style_names.txt
caffe_root = '/home/zju/wlj/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import os

os.chdir(caffe_root)
style_labels = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
                'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
                'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
style_label_file = '/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/style_names.txt'
with open(style_label_file, 'w') as file:
    for label in style_labels:
        file.write(label + '\n')

# 准备train.txt和test.txt
# !dos2unix /home/zju/wlj/land_use_cnn/data/create_filelist.sh
# !bash /home/zju/wlj/land_use_cnn/data/create_filelist.sh
# !head -5 /home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/test.txt

# 生成均值文件binaryproto
# !dos2unix /home/zju/wlj/land_use_cnn/data/create_lmdb.sh
# !bash /home/zju/wlj/land_use_cnn/data/create_lmdb.sh
# !dos2unix /home/zju/wlj/land_use_cnn/data/create_mean.sh
# !bash /home/zju/wlj/land_use_cnn/data/create_mean.sh

# 均值文件转npy
caffe_root = '/home/zju/wlj/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys

sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

MEAN_PROTO_PATH = '/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/mean.binaryproto'  # 待转换的pb格式图像均值文件路径
MEAN_NPY_PATH = '/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse/mean.npy'  # 转换后的numpy格式图像均值文件路径

blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
data = open(MEAN_PROTO_PATH, 'rb').read()  # 读入mean.binaryproto文件内容
blob.ParseFromString(data)  # 解析文件内容到blob

array = np.array(
    caffe.io.blobproto_to_array(blob))  # 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
mean_npy = array[0]  # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
np.save(MEAN_NPY_PATH, mean_npy)
