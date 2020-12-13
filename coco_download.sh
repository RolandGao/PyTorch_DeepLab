mkdir coco
cd coco
mkdir images
cd images

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip -q train2017.zip
unzip -q val2017.zip

rm train2017.zip
rm val2017.zip

cd ..

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip
