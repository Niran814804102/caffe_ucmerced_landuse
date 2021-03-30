#!/usr/bin/env sh
TOOL=/root/caffe-master/build/tools
MY=/root/caffe-master/models/finetune_UCMerced_LandUse

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

