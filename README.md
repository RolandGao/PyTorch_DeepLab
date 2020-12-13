# PyTorch_DeepLab

Currently, the code supports DeepLabv3+ with many common backbones, such as Mobilenetv2, Mobilenetv3, Resnet, Resnetv2, XceptionAligned, Regnet, EfficientNet, and many more, thanks to the package [timm](https://github.com/rwightman/pytorch-image-models). The code supports 3 datasets, namely PascalVoc, Coco, and Cityscapes.


I trained a few models on Cityscapes and PascalVoc, and will release the weights soon.

# Results
PascalVoc with Resnet50v2: 77.1 mIOU


# To train a model yourself

Download one of the three datasets, change dataset_path, save_path, and num_classes in train.py if necessary, and run with the command "python train.py".
