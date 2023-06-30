"""
Here we put the methods to augment and randomize the dataset samples

TODO: Replace ball augmentation.
"""

import numpy as np
import torch
import torchvision.transforms.functional as F

from lib.logger import print_
from CONFIG import AUGMENTATIONS, SPATIAL_AUGMENTATIONS


class Augmentator:
    """
    Module for augmenting an image from the dataset

    Args:
    -----
    augment_params: dictionary
        augmentations chunk from the experiment parameters
    """

    def __init__(self, augment_params):
        """ Initializer of the augmentator """
        self.augmentations = []
        augmentations = augment_params["augmentations"]
        for augment_name in augmentations:
            if augment_name == "Horizontal flip":
                augment = RandomHorizontalFlip(mirror_prob=0.5)
            elif augment_name == "Scale":
                augment = Scale(min_size=0.75, max_size=1.25)
            elif augment_name == "Rotation":
                degrees = augment_params.get("degrees", 20)
                augment = RandomRotation(degrees=degrees)
            elif augment_name == "Brightness":
                mean = augment_params.get("mean_brightness", 1)
                std = augment_params.get("std_brightness", 0.6)
                augment = AdjustBrightness(mean=mean, std=std)
            elif augment_name == "Contrast":
                mean = augment_params.get("mean_contrast", 1)
                std = augment_params.get("std_contrast", 0.3)
                augment = AdjustContrast(mean=mean, std=std)
            elif augment_name == "Saturation":
                mean = augment_params.get("mean_saturation", 1)
                std = augment_params.get("std_saturation", 0.6)
                augment = AdjustSaturation(mean=mean, std=std)
            elif augment_name == "Hue":
                mean = augment_params.get("mean_hue", 0.05)
                std = augment_params.get("std_hue", 0.05)
                augment = AdjustHue(mean=mean, std=std)
            elif augment_name == "Blur":
                prob = augment_params.get("blur_prob", 0.5)
                blur_kernel = augment_params.get("blur_kernel", (3, 7))
                sigma = augment_params.get("blur_sigma", (1, 10))
                augment = MotionBlur(blur_kernel=blur_kernel, prob=prob, sigma=sigma)
            else:
                raise ValueError(f"Augmentation {augment_name} not available. Use one of {AUGMENTATIONS}.")
            self.augmentations.append((augment_name, augment))

        print_("Using Augmentations:")
        for aug in self.augmentations:
            print_(f"    {aug[1]}")
        return

    def augment_img(self, x, use_params=None):
        """ Applying list of augmentations to one image """
        use_params = {} if use_params is None else use_params
        params = {}
        for augment_name, augmentation in self.augmentations:
            x = augmentation(x, **use_params)
            params = {**params, **augmentation.get_params()}
        return x, params

    def augment_annotation(self, x, use_params=None):
        """ Applying list of augmentations to one annotation. """
        use_params = {} if use_params is None else use_params
        params = {}
        for augment_name, augmentation in self.augmentations:
            if augment_name not in SPATIAL_AUGMENTATIONS:
                continue
            x = augmentation(x, **use_params)
            params = {**params, **augmentation.get_params()}
        return x, params

    def __call__(self, img, label=None):
        """ Applying random augmentations to one image and its corresponding label"""
        augment_img, params = self.augment_img(img)
        if label is not None:
            augment_label, _ = self.augment_annotation(label, params)
            return (augment_img, augment_label)
        else:
            return augment_img

    def __repr__(self):
        """ For displaying nicely """
        message = f"Compose of {len(self.tf_list)} transforms:\n"
        for tf in self.tf_list:
            message += f"    {self.__class__.__name__}\n"
        message = message[:-1]  # removing last line break
        return message


class Augmentation:
    """
    Base class for self-implemented augmentations

    Args:
    -----
    params: list
        list of parameters used for the augmentation
    """

    def __init__(self, params):
        """ Module initializer """
        self.params = params
        self.log = None
        return

    def __call__(self, x, **kwargs):
        """ Auctually augmenting one image"""
        raise NotImplementedError("Base class does not implement augmentations")

    def log_params(self, values):
        """ Saving the exact sampled value for the current augment """
        assert len(values) == len(self.params), \
            f"ERROR! Length of value ({len(values)}) and params ({len(self.params)}) do not match"
        self.log = {p: v for p, v in zip(self.params, values)}
        return

    def get_params(self):
        """ Fetching parameters and values """
        return self.log


class RandomHorizontalFlip(Augmentation):
    """
    Horizontally mirroring an image given a certain probability
    """

    def __init__(self, mirror_prob=0.5):
        """ Augmentation initializer """
        super().__init__(params=["mirror_prob"])
        self.mirror_prob = mirror_prob

    def __call__(self, x, **kwargs):
        """ Mirroring the image """
        mirror = kwargs.get("mirror_prob", np.random.rand() < self.mirror_prob)
        self.log_params([mirror])
        if mirror:
            x_augment = F.hflip(x)
        else:
            x_augment = x
        return x_augment

    def __repr__(self):
        """ String representation """
        str = f"RandomHorizontalFlip(mirror_prob={self.mirror_prob})"
        return str


class Scale(Augmentation):
    """ 
    Random scaling of the image and annotation
    """
    
    def __init__(self, min_size=0.75, max_size=1.2):
        """ Augmentaiton initializer """
        super().__init__(params=["scale_factor"])
        self.min_size = min_size
        self.max_size = max_size
        
        def __call__(self, x, **kwargs):
            """ Scaling the image """
            min_size = kwargs.get("min_size")
            max_size = kwargs.get("min_size")
            scale_factor = min_size + np.random.random() * (max_size - min_size)
            self.log_params([scale_factor])
            H, W = int(round(x.shape[-2] * scale_factor)), int(round(x.shape[-1] * scale_factor))
            x_augment = F.resize(img=x, size=(H, W))
            return x_augment

    def __repr__(self):
        """ String representation """
        str = f"Scale(range=[{self.min_size}, {self.max_size}])"
        return str


class RandomRotation(Augmentation):
    """
    Rotating an image for certaing angle
    """

    def __init__(self, degrees=20):
        """ Augmentation initializer """
        super().__init__(params=["degrees"])
        self.degrees = degrees

    def __call__(self, x, **kwargs):
        """ Rotating the image by a random sample angle """
        random_angle = kwargs.get("degrees", (np.random.rand() * self.degrees*2) - self.degrees)
        self.log_params([random_angle])
        x_augment = F.rotate(x, random_angle)
        return x_augment

    def __repr__(self):
        """ String representation """
        str = f"RandomRotation(degrees={self.degrees})"
        return str


class AdjustContrast(Augmentation):
    """
    Adjusting the contrat of an image
    """

    def __init__(self, mean=1., std=0.3, min=0.4, max=1.6):
        """ Initializer """
        super().__init__(params=["contrast"])
        self.std = std
        self.mean = mean
        self.get_contrast = lambda: (self.mean + torch.randn(1) * self.std).clamp(min, max)
        return

    def __call__(self, img, **kwargs):
        """ Actually adding noise to the image """
        contrast = kwargs.get("contrast", self.get_contrast())
        self.log_params([contrast])
        augm_img = F.adjust_contrast(img=img, contrast_factor=contrast)
        return augm_img

    def __repr__(self):
        """ String representation """
        str = f"AdjustContrast(mean={self.mean}, std={self.std})"
        return str


class AdjustSaturation(Augmentation):
    """ Adjusting the saturation of an image """

    def __init__(self, mean=1., std=0.6, min=0.4, max=1.6):
        """ Initializer """
        super().__init__(params=["saturation"])
        self.std = std
        self.mean = mean
        self.get_saturation = lambda: (self.mean + torch.randn(1) * self.std).clamp(min, max)
        return

    def __call__(self, img, **kwargs):
        """ Actually adding noise to the image """
        saturation = kwargs.get("saturation", self.get_saturation())
        self.log_params([saturation])
        augm_img = F.adjust_saturation(img=img, saturation_factor=saturation)
        return augm_img

    def __repr__(self):
        """ String representation """
        str = f"AdjustSaturation(mean={self.mean}, std={self.std})"
        return str


class AdjustBrightness(Augmentation):
    """
    Adjusting the brightness of an image
    """

    def __init__(self, mean=1., std=0.6, min=0.1, max=1.9):
        """ Initializer """
        super().__init__(params=["brightness"])
        self.std = std
        self.mean = mean
        self.get_brightness = lambda: (self.mean + torch.randn(1) * self.std).clamp(min, max)
        return

    def __call__(self, img, **kwargs):
        """ Actually adding noise to the image """
        brightness = kwargs.get("brightness", self.get_brightness())
        self.log_params([brightness])
        augm_img = F.adjust_brightness(img=img, brightness_factor=brightness)
        return augm_img

    def __repr__(self):
        """ String representation """
        str = f"AdjustBrightness(mean={self.mean}, std={self.std})"
        return str


class AdjustHue(Augmentation):
    """ Adjusting the hue of an image """

    def __init__(self, mean=0.05, std=0.05, min=-0.1, max=0.3):
        """ Initializer """
        super().__init__(params=["hue"])
        self.std = std
        self.mean = mean
        self.get_hue = lambda: (self.mean + torch.randn(1) * self.std).clamp(min, max)
        return

    def __call__(self, img, **kwargs):
        """ Actually changing hue """
        hue = kwargs.get("hue", self.get_hue())
        self.log_params([hue])
        augm_img = F.adjust_hue(img=img, hue_factor=hue)
        return augm_img

    def __repr__(self):
        """ String representation """
        str = f"AdjustHue(mean={self.mean}, std={self.std})"
        return str


class AddNoise(Augmentation):
    """ Custom augmentation to add some random Gaussiabn Noise to an Image """

    def __init__(self, mean=0., std=0.3):
        """ Initializer """
        super().__init__(params=["mean", "std"])
        self.std = std
        self.mean = mean
        self.rand_std = lambda: torch.rand(1)*self.std
        return

    def __call__(self, img, **kwargs):
        """ Actually adding noise to the image """
        random_std = self.rand_std()
        self.log_params([0.0, random_std])
        noise = torch.randn(img.shape) * random_std + self.mean
        noisy_img = img + noise.to(img.device)
        noisy_img = noisy_img.clamp(0,1)
        return noisy_img

    def __repr__(self):
        """ String representation """
        str = f"AddNoise(mean={self.mean}, std={self.std})"
        return str


class MotionBlur(Augmentation):
    """ Custom augmentation to add some motion Blur to an Image """

    def __init__(self, blur_kernel=(3, 8), prob=0.5, sigma=(1, 10)):
        """ Initializer """
        super().__init__(params=["blur_kernel", "blur_active", "sigma"])
        self.blur_kernel = blur_kernel
        self.prob = prob
        self.sigma = sigma

        self.blur = lambda: np.random.uniform(0, 1) < 0.5
        self.get_kernel = lambda: np.random.randint(*blur_kernel) * 2 - 1
        self.get_sigma = lambda: np.random.uniform(*sigma)
        return

    def __call__(self, img, **kwargs):
        """ Actually adding noise to the image """
        blur = kwargs.get("blur_active", self.blur())
        if blur:
            kernel = kwargs.get("blur_kernel", (self.get_kernel(), self.get_kernel()))
            sigma = kwargs.get("sigma", (self.get_sigma(), self.get_sigma()))
            blurred_img = F.gaussian_blur(img=img, kernel_size=kernel, sigma=sigma)
        else:
            blurred_img = img
            kernel, sigma = None, None
        self.log_params([kernel, blur, sigma])
        return blurred_img

    def __repr__(self):
        """ String representation """
        str = f"MotionBlur(blur_kernel={self.blur_kernel}, prob={self.prob}, sigma={self.sigma})"
        return str


#
