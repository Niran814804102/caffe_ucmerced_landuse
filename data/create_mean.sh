#!/usr/bin/env sh
TOOL=/home/zju/wlj/caffe-master/build/tools
MY=/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse
echo "Create train mean.."
rm -rf $MY/mean.binaryproto
$TOOL/compute_image_mean $MY/img_train_lmdb $MY/mean.binaryproto
echo "All Done.."

