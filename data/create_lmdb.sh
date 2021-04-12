#!/usr/bin/env sh
TOOL=/home/zju/wlj/caffe-master/build/tools
MY=/home/zju/wlj/land_use_cnn/data/UCMerced_LandUse
echo "Create train lmdb.."
rm -rf $MY/img_train_lmdb
$TOOL/convert_imageset \
--shuffle \
--resize_height=256 \
--resize_width=256 \
/ \
$MY/train.txt \
$MY/img_train_lmdb

echo "Create test lmdb.."
rm -rf $MY/img_test_lmdb
$TOOL/convert_imageset \
--shuffle \
--resize_width=256 \
--resize_height=256 \
/ \
$MY/test.txt \
$MY/img_test_lmdb

echo "All Done.."

