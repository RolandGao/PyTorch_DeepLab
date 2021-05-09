import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from model import Deeplab3P
import time
from data import get_cityscapes,get_pascal_voc
from cityscapes import Cityscapes

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def get_colors():
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.arange(255).view(-1, 1) * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors
def get_colors_cityscapes():
    colors=np.zeros((256,3))
    colors[255]=[255,255,255]
    for c in Cityscapes.classes:
        if 0<=c.train_id<=18:
            colors[c.train_id]=c.color
    return colors.astype("uint8")


def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def show_mask(images):
    colors=get_colors()
    r = Image.fromarray(images.byte().cpu().numpy())
    r.putpalette(colors)
    plt.imshow(r)
def show_cityscapes_mask(images):
    colors=get_colors_cityscapes()
    r = Image.fromarray(images.byte().cpu().numpy())
    r.putpalette(colors)
    plt.imshow(r)

def display(data_loader,show_mask,num_images=5,skip=4,images_per_line=6):
    images_so_far = 0
    fig = plt.figure(figsize=(6, 4))
    num_rows=int(np.ceil(num_images/images_per_line))
    data_loader = iter(data_loader)
    for _ in range(skip):
        next(data_loader)
    for images, targets in data_loader:
        for image, target in zip(images, targets):
            print(image.size(), target.size())
            plt.subplot(num_rows, 2*images_per_line, images_so_far + 1)
            plt.axis('off')
            show_image(image)

            plt.subplot(num_rows, 2*images_per_line, images_so_far + 2)
            plt.axis('off')
            show_mask(target)

            images_so_far += 2
            if images_so_far == 2 * num_images:
                plt.tight_layout()
                plt.show()
                return
    plt.tight_layout()
    plt.show()
def show(model,data_loader,device,show_mask,num_images=5,skip=4,images_per_line=2):
    images_so_far=0
    model.eval()
    num_rows = int(np.ceil(num_images / images_per_line))
    fig=plt.figure(figsize=(8,4))
    data_loader=iter(data_loader)
    for _ in range(skip):
        next(data_loader)
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            start=time.time()
            outputs = model(images)
            end=time.time()
            print(end-start)
            for image,target,output in zip(images,targets,outputs):
                output = output.argmax(0)
                print(image.size(),target.size(),output.size())
                plt.subplot(num_rows, 3*images_per_line, images_so_far+1)
                plt.axis('off')
                show_image(image)

                plt.subplot(num_rows, 3*images_per_line, images_so_far+2)
                plt.axis('off')
                show_mask(target)

                plt.subplot(num_rows,3*images_per_line,images_so_far+3)
                plt.axis('off')
                show_mask(output)

                images_so_far+=3
                if images_so_far==3*num_images:
                    plt.tight_layout()
                    plt.show()
                    return
    plt.tight_layout()
    plt.show()

def show_cityscapes():
    num_images=16
    images_per_line=4
    skip=0
    _,data_loader=get_cityscapes("cityscapes_dataset",16,train_size=481,val_size=513)
    display(data_loader,show_cityscapes_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)

if __name__=="__main__":
    show_cityscapes()

    #_,data_loader=get_pascal_voc("pascal_voc_dataset",16,train_size=385,val_size=385)

    #pretrained_path='checkpoints/voc_resnet50d_noise'
    #model=torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
    # model=Deeplab3P(name='resnet50d',num_classes=num_classes,pretrained=pretrained_path,sc=True).to(
    #     device)

    #show(model,data_loader,device,show_mask,num_images=num_images,skip=skip,images_per_line=images_per_line)
