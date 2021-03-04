import transforms as T
from data_utils import *
from cityscapes import Cityscapes
from voc12 import Voc12Segmentation
from coco_utils import get_coco_dataset

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

def build_transforms2(is_train,size,crop_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    pad_value = tuple([int(v * 255) for v in mean])
    ignore_label = 255
    transforms=[]
    min_scale=1
    max_scale=1
    if is_train:
        min_scale=0.5
        max_scale=2
    transforms.append(T.RandomResize(int(min_scale*size),int(max_scale*size)))
    transforms.append(T.ColorJitter(0.5,0.5,(0.5,2),0.05))
    transforms.append(T.AddNoise(20))
    transforms.append(T.RandomRotation((-20,20),mean=pad_value,ignore_value=0))
    if is_train:
        transforms.append(
        T.RandomCrop(
            crop_size,crop_size,
            pad_value,
            ignore_label,
            random_pad=is_train
        ))
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)

def get_cityscapes(root,batch_size=16,val_size=513,train_size=481):
    train=Cityscapes(root,split="train",target_type="semantic",transforms=build_transforms2(True,val_size,train_size))
    val=Cityscapes(root,split="val",target_type="semantic",transforms=build_transforms2(False,val_size,train_size))
    train_loader = get_dataloader_train(train, batch_size)
    val_loader = get_dataloader_val(val)
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader,val_loader
def get_coco(root,batch_size=16,val_size=513,train_size=481):
    train=get_coco_dataset(root, "train", build_transforms2(True,val_size,train_size))
    val=get_coco_dataset(root, "val", build_transforms2(False,val_size,train_size))
    train_loader = get_dataloader_train(train, batch_size)
    val_loader = get_dataloader_val(val)
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader
def get_pascal_voc(root,batch_size=16,val_size=513,train_size=481):
    download=False
    train = Voc12Segmentation(root, 'train_aug', build_transforms2(True,val_size,train_size),
                                download)
    val = Voc12Segmentation(root, 'val', build_transforms2(False,val_size,train_size),
                                     download)
    train_loader = get_dataloader_train(train, batch_size)
    val_loader = get_dataloader_val(val)
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader

if __name__ == '__main__':
    import torchvision
    import torch.utils.data
    import timm
    print(timm.list_models())
