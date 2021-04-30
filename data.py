import transforms as T
from data_utils import *
from cityscapes import Cityscapes
from voc12 import Voc12Segmentation
from coco_utils import get_coco_dataset

def build_transforms(is_train, size, crop_size,mode="baseline"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
    ignore_value = 255
    transforms=[]
    min_scale=1
    max_scale=1
    if is_train:
        min_scale=0.5
        max_scale=2
    transforms.append(T.RandomResize(int(min_scale*size),int(max_scale*size)))
    if is_train:
        if mode=="baseline":
            pass
        elif mode=="randaug":
            transforms.append(T.RandAugment(2,1/3,prob=1.0,fill=fill,ignore_value=ignore_value))
        elif mode=="custom1":
            transforms.append(T.ColorJitter(0.5,0.5,(0.5,2),0.05))
            transforms.append(T.AddNoise(10))
            transforms.append(T.RandomRotation((-10,10), mean=fill, ignore_value=0))
        else:
            raise NotImplementedError()
        transforms.append(
        T.RandomCrop(
            crop_size,crop_size,
            fill,
            ignore_value,
            random_pad=is_train
        ))
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)

def get_cityscapes(root,batch_size=16,val_size=513,train_size=481,mode="baseline"):
    train=Cityscapes(root, split="train", target_type="semantic", transforms=build_transforms(True, val_size, train_size,mode))
    val=Cityscapes(root, split="val", target_type="semantic", transforms=build_transforms(False, val_size, train_size,mode))
    train_loader = get_dataloader_train(train, batch_size)
    val_loader = get_dataloader_val(val)
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader,val_loader
def get_coco(root,batch_size=16,val_size=513,train_size=481,mode="baseline"):
    train=get_coco_dataset(root, "train", build_transforms(True, val_size, train_size,mode))
    val=get_coco_dataset(root, "val", build_transforms(False, val_size, train_size,mode))
    train_loader = get_dataloader_train(train, batch_size)
    val_loader = get_dataloader_val(val)
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader
def get_pascal_voc(root,batch_size=16,val_size=513,train_size=481,mode="baseline"):
    download=False
    train = Voc12Segmentation(root, 'train_aug', build_transforms(True, val_size, train_size,mode),
                              download)
    val = Voc12Segmentation(root, 'val', build_transforms(False, val_size, train_size,mode),
                            download)
    train_loader = get_dataloader_train(train, batch_size)
    val_loader = get_dataloader_val(val)
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader=get_pascal_voc("pascal_voc_dataset")
    print(iter(train_loader).__next__())
