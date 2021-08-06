import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode 

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
        return image, target

class Resize(object):
    def __init__(self, size, image_only=False):
        self.size = size
        self.image_only = image_only

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        if not self.image_only:
            target = F.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class LabelRemap(object):
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, image, target):
        target_np = np.asarray(target, dtype='uint8')
        target_cp = target_np.copy()
        for k, v in self.mapping.items():
            target_cp[target_np == k] = v
        target = Image.fromarray(np.uint8(target_cp))
        return image, target


class RandomCrop(object):
    def __init__(self, size, ignore_label=255):
        self.size = size
        self.ignore_label = ignore_label

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=self.ignore_label)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __init__(self, img_mode='BGR'):
        self.img_mode = img_mode
        assert(self.img_mode == 'BGR' or self.img_mode == 'RGB')

    def __call__(self, image, target):
        if self.img_mode == 'RGB':
            image = F.to_tensor(image)
        else:
            image = np.asarray(image, np.float32)
            # change image to BGR
            image = image[:, :, ::-1].copy() 
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)

        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ColorJittering(object):
    def __init__(self, b=0, c=0, s=0, h=0):
        self.t = T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

    def __call__(self, image, target):
        image = self.t(image)
        return image, target

