#!/usr/bin/env sh
DATA=/root/caffe-master/data/UCMerced_LandUse
MY=/root/caffe-master/models/finetune_UCMerced_LandUse
LABELS=('agricultural' 'airplane' 'baseballdiamond' 'beach' 'buildings' 'chaparral' 'denseresidential' 'forest' 'freeway' 'golfcourse' 'harbor' 'intersection' 'mediumresidential' 'mobilehomepark' 'overpass' 'parkinglot' 'river' 'runway' 'sparseresidential' 'storagetanks' 'tenniscourt')

echo "Create train.txt..."
rm -rf $MY/train.txt

for ((i=0;i<${#LABELS[*]};i++))
do
    find $DATA/train -name ${LABELS[$i]}*.jpg | sed "s/$/ ${i}/" >> $MY/train.txt
done


echo "Create test.txt..."
rm -rf $MY/test.txt

for ((i=0;i<${#LABELS[*]};i++))
do
    find $DATA/test -name ${LABELS[$i]}*.jpg | sed "s/$/ ${i}/" >> $MY/test.txt
done
echo "All done"
