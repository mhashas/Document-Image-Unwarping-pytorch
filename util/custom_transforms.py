import types
import random
import torch

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import functional as F
import numpy as np

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        #assert image.size == label.size

        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomCrop(object):
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, image, label):
        if self.padding > 0:
            image = ImageOps.expand(image, border=self.padding, fill=0)
            label = ImageOps.expand(label, border=self.padding, fill=0)
        # @TODO RADU
        #assert image.size == label.size
        w, h = image.size
        tw, th = self.size

        if w == tw and h == th:
            return image, label
        if w < tw or h < th:
            return image.resize((tw, th), Image.BILINEAR), label.resize((tw, th), Image.BILINEAR)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        cropped_images, cropped_labels = image.crop((x1, y1, x1 + tw, y1 + th)), label.crop((x1, y1, x1 + tw, y1 + th))
        return cropped_images, cropped_labels

class RandomHorizontallyFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label

class Scale(object):
    def __init__(self, size, label_resize_type=Image.BILINEAR):
        self.size = size
        self.label_resize_type = label_resize_type

    def __call__(self, image, label):
        assert image.size == label.size
        w, h = image.size

        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return image, label
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return image.resize((ow, oh), Image.BILINEAR), label.resize((ow, oh), self.label_resize_type)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return image.resize((ow, oh), Image.BILINEAR), label.resize((ow, oh), self.label_resize_type)

class Resize(object):
    def __init__(self, size, label_resize_type=Image.BILINEAR):
        self.size = size
        self.label_resize_type = label_resize_type

    def __call__(self, image, label):
        #assert image.size == label.size
        w, h = image.size
        w_label, h_label = label.size

        if not ((w >= h and w == self.size[0]) or (h >= w and h == self.size[1])):
            image = image.resize(self.size, Image.BILINEAR)

        if not ((w_label >= h_label and w_label == self.size[0]) or (h_label >= w_label and h_label == self.size[1])):
            label = label.resize(self.size, self.label_resize_type)

        return image, label

class RandomGaussianBlur(object):
    def __call__(self, image, label=None):
        if random.random() < 0.5:
            radius = random.random()
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        if label is not None:
            return image, label
        return image

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        return transforms

    @staticmethod
    def forward_transforms(image, transforms):
        for transform in transforms:
            image = transform(image)

        return image

    def __call__(self, image, label):
        """
        Args:
            images (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transforms = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return self.forward_transforms(image, transforms), label