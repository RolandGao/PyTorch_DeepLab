import random
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import torch

class Compose(object):
    """
    Composes a sequence of transforms.
    Arguments:
        transforms: A list of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    """
    Normalizes image by mean and std.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomScale(object):
    """
    Applies random scale augmentation.
    Arguments:
        min_scale: Minimum scale value.
        max_scale: Maximum scale value.
        scale_step_size: The step size from minimum to maximum value.
    """
    def __init__(self, min_scale, max_scale, scale_step_size):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

    @staticmethod
    def get_random_scale(min_scale_factor, max_scale_factor, step_size):
        """Gets a random scale value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        Raises:
            ValueError: min_scale_factor has unexpected value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')

        if min_scale_factor == max_scale_factor:
            return min_scale_factor

        # When step_size = 0, we sample the value uniformly from [min, max).
        if step_size == 0:
            return random.uniform(min_scale_factor, max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
        np.random.shuffle(scale_factors)
        return scale_factors[0]

    def __call__(self, image, label):
        scale = self.get_random_scale(self.min_scale, self.max_scale, self.scale_step_size)
        img_w, img_h = image.size
        img_w,img_h=int(img_w*scale),int(img_h*scale)
        image=F.resize(image,[img_h,img_w])
        label=F.resize(label,[img_h,img_w],interpolation=Image.NEAREST)
        return image,label


class RandomCrop(object):
    def __init__(self, crop_h, crop_w, pad_value, ignore_label, random_pad):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_label = ignore_label
        self.random_pad = random_pad

    def __call__(self, image, label):
        img_w,img_h=image.size
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
            image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.pad_value)
            label= F.pad(label, (pad_left, pad_top, pad_right, pad_bottom), fill=self.ignore_label)

        crop_params = T.RandomCrop.get_params(image, (self.crop_h, self.crop_w))
        image = F.crop(image, *crop_params)
        label = F.crop(label, *crop_params)
        return image,label


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target
