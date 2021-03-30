#!/usr/bin/env sh
TOOL=/root/caffe-master/build/tools
MY=/root/caffe-master/models/finetune_UCMerced_LandUse

echo "Create train mean.."
rm -rf $MY/mean.binaryproto
$TOOL/compute_image_mean $MY/img_train_lmdb $MY/mean.binaryproto
echo "All Done.."

