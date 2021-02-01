import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from data import get_cityscapes,get_pascal_voc
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def get_colors():
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.arange(255).view(-1, 1) * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

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

if __name__=="__main__":
    data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",16)
    display(data_loader,show_mask,num_images=6,skip=0,images_per_line=3)
