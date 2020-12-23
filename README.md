# PyTorch_DeepLab

Currently, the code supports DeepLabv3+ with many common backbones, such as Mobilenetv2, Mobilenetv3, Resnet, Resnetv2, XceptionAligned, Regnet, EfficientNet, and many more, thanks to the package [timm](https://github.com/rwightman/pytorch-image-models). The code supports 3 datasets, namely PascalVoc, Coco, and Cityscapes.


I trained a few models on Cityscapes and PascalVoc, and will release the weights soon.

## Results

Using separable convolution in the decoder 
reduces model size and the number of flops, 
but increases the memory requirement by 1 GB during training.

#### PascalVoc
To use the weights, click the link, and instantiate an object like the line below, 
changing the name, sc("separable convolution"), and the path to the pretrained weights that you just downloaded.

```
model=Deeplab3P(name='regnetx_040',num_classes=21,
sc=False,pretrained=pretrained_path).to(device)
```
name | separable convolution | mIOU | weights
--- | --- | --- | ---
resnet50d | yes | 77.1 | [link](https://github.com/RolandGao/PyTorch_DeepLab/releases/download/v1.0-alpha/voc_resnet50d)
regnetx_040 | yes | 77.0 | [link](https://github.com/RolandGao/PyTorch_DeepLab/releases/download/v1.0-alpha/voc_regnetx40)
regnety_040 | no | 78.6 | [link](https://github.com/RolandGao/PyTorch_DeepLab/releases/download/v1.0-alpha/voc_regnety40)
mobilenetv2 | no | 72.8 | [link](https://github.com/RolandGao/PyTorch_DeepLab/releases/download/v1.0-alpha/voc_mobilenetv2)

## Installation
After cloning the repository, run the following command to install all dependencies.
pip install -r requirements.txt

## Datasets
#### COCO
run the command
```shell
sh coco_download.sh
```
We use the 21 classes that intersect PascalVoc's.

#### Cityscapes
go to https://www.cityscapes-dataset.com, create an account, and download 
gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip.
You can delete the test images to save some space if you don't want to submit to the competition.
Name the directory cityscapes_dataset.
Make sure that you have downloaded the required python packages and run
```
CITYSCAPES_DATASET=cityscapes_dataset csCreateTrainIdLabelImgs
```
There are 19 classes.

#### PascalVoc
Download the original dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).

Then download the augmented dataset [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0),
and create a text file named train_aug.txt with [this content](https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt).

Place train_aug.txt in VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt

Place SegmentationClassAug directory in VOCdevkit/VOC2012/SegmentationClassAug

There are 21 claases.

Credits to https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/


#### Once you have downloaded the dataset
do one of the following three lines in train.py
```
data_loader, data_loader_test=get_coco(root,batch_size=16)
data_loader, data_loader_test=get_pascal_voc(root,batch_size=16)
data_loader, data_loader_test=get_cityscapes(root,batch_size=16)
```
where the root is usually "." or the top level directory name of the dataset.

## To train a model yourself
Download one of the three datasets, change save_path, and num_classes in train.py if necessary, and run the command 
```
python train.py
```

## To resume training
In train.py, set resume=True, and change the resume_path to the save_path of your last train session.
