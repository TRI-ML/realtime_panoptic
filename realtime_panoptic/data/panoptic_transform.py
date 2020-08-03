# Copyright 2020 Toyota Research Institute.  All rights reserved.
import random

import torchvision.transforms as torchvision_transforms
from PIL import Image
from torchvision.transforms import functional as F


class Compose:
    """Compose as set of data transform operations

    Parameters:
    -----------
    transforms: list
        A list of transform operations.

    Returns:
    -------
    data: Dict
        The final output data after the given set of transforms.  
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize:
    """Resize operation on panoptic data.
    The input data will be resized by randomly choose a minimum side length from
    a given set with the maximum side capped by a given length.

    Parameters:
    ----------
    min_size: list or tuple
        A list of size to be chosen for image minimum side.

    max_size: int
        Maximum side length of the processed image

    """
    def __init__(self, min_size, max_size, is_train=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size, )
        self.min_size = min_size
        self.max_size = max_size
        self.is_train = is_train

    # modified from torchvision to add support for max size
    # NOTE: this method will always try to make the smaller size match ``self.min_size``
    # so in the case of Vistas, this will mean evaluating at a significantly lower resolution
    # than the original image for some images
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, data):
        size = self.get_size(data["image"].size)
        data["image"] = F.resize(data["image"], size)
        if self.is_train:
            if "segmentation_target" in data:
                data["segmentation_target"] = F.resize(
                    data["segmentation_target"],
                    size,
                    interpolation=Image.NEAREST)
            if "detection_target" in data:
                data["detection_target"] = data["detection_target"].resize(
                    data["image"].size)
        return data


class RandomHorizontalFlip:
    """Randomly Flip the input data with given probability.

    Parameters:
    ----------
    prob: float
        A probability to flip the data in [0,1].
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data["image"] = F.hflip(data["image"])
            if "detection_target" in data:
                data["detection_target"] = data["detection_target"].transpose(
                    0)
            if "segmentation_target" in data:
                data["segmentation_target"] = F.hflip(
                    data["segmentation_target"])
        return data


class ToTensor:
    """Convert the input data to Tensor.
    """
    def __call__(self, data):
        data["image"] = F.to_tensor(data["image"])
        if "segmentation_target" in data:
            data["segmentation_target"] = F.to_tensor(
                data["segmentation_target"])
        return data


class ColorJitter:
    """Apply color jittering to input image.
    """
    def __call__(self, data):
        data["image"] = torchvision_transforms.ColorJitter().__call__(
            data["image"])
        return data


class Normalize:
    """Normalize the input image with options of RGB/BGR converting.

    Parameters:
    ----------
    mean: list
        Mean value for the 3 image channels.

    std: list
        Standard deviation for the 3 image channels.

    to_bgr255: bool
        If true, the default image come in with rgb channel and [0,1] scale.
        it will be converted into bgr with [0,255] scale.
    """
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, data):
        if self.to_bgr255:
            data["image"] = data["image"][[2, 1, 0]] * 255
            if "segmentation_target" in data:
                data["segmentation_target"] = (
                    data["segmentation_target"] * 255).long()
        data["image"] = F.normalize(
            data["image"], mean=self.mean, std=self.std)
        return data


class RandomCrop:
    """Randomly Crop in input panoptic data. 

    Parameters:
    ----------
    crop_size: tuple
        Desired crop size of the data. 
    """
    def __init__(self, crop_size):
        # A couple of safety checks
        assert isinstance(crop_size, tuple)
        self.crop_size = crop_size
        self.crop = torchvision_transforms.RandomCrop(crop_size)

    def __call__(self, data):
        if len(self.crop_size) <= 1:
            return data

        # If image size is smaller than crop size,
        # resize both image and target to at least crop size.
        if self.crop_size[0] > data["image"].size[0] or self.crop_size[
                1] > data["image"].size[1]:
            print("Image will be resized before cropping. {},{}".format(
                self.crop_size, data["image"].size))
            resize_func = Resize(
                max(self.crop_size),
                round(
                    max(data["image"].size) * max(self.crop_size) / min(
                        data["image"].size)))
            data = resize_func(data)

        if "detection_target" not in data:

            image_width, image_height = data["image"].size
            crop_width, crop_height = self.crop_size
            assert image_width >= crop_width and image_height >= crop_height

            left = 0
            if image_width > crop_width:
                left = random.randint(0, image_width - crop_width)
            top = 0
            if image_height > crop_height:
                top = random.randint(0, image_height - crop_height)

            data["image"] = data["image"].crop((left, top, left + crop_width,
                                                top + crop_height))
            if "segmentation_target" in data:
                data["segmentation_target"] = data["segmentation_target"].crop(
                    (left, top, left + crop_width, top + crop_height))

        else:
            # We always crop an area that contains at least one instance.
            # TODO: We are making an assumption here that data are filtered to only include
            # non-empty training samples. So the while loop will not be a dead lock. Need to 
            # improve the efficiency of this part. 
            while True:
                # continuously try till there's instance inside it.
                w, h = data["image"].size
                if w <= self.crop_size[0] or h <= self.crop_size[1]:
                    break
                # y, x, h, w
                top, left, crop_height, crop_width = self.crop.get_params(
                    data["image"], (self.crop_size[1], self.crop_size[0]))

                image_cropped = F.crop(data["image"], top, left, crop_height,
                                       crop_width)

                detection_target_cropped = data[
                    "detection_target"].augmentation_crop(
                        top, left, crop_height, crop_width)

                if detection_target_cropped is not None:
                    data["image"] = image_cropped
                    data["detection_target"] = detection_target_cropped

                    # Once ``detection_target`` gets properly cropped, then crop
                    # ``segmentation_target`` using the same parameters
                    if "segmentation_target" in data:
                        data["segmentation_target"] = F.crop(
                            data["segmentation_target"], top, left,
                            crop_height, crop_width)
                    break
        return data