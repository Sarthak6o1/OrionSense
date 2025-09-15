# utils/transforms.py
from PIL import Image
import random
import torchvision.transforms.functional as F
import torch

class Compose(object):
    """Compose transforms that accept (image_pil, target_dict) and return (image_pil, target_dict)."""
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    """Convert PIL image to torch tensor (C,H,W). Does not change target."""
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Resize(object):
    """Resize PIL image to given (width, height) and scale bounding boxes in target accordingly.
       size may be (W,H) or int (shorter side)."""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (W,H)
    def __call__(self, image, target):
        # original size
        ow, oh = image.size
        nw, nh = self.size
        image = image.resize((nw, nh), resample=Image.BILINEAR)
        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes'].clone().float()
            # scale x coordinates by nw/ow, y coords by nh/oh
            boxes[:, [0,2]] = boxes[:, [0,2]] * (nw / ow)
            boxes[:, [1,3]] = boxes[:, [1,3]] * (nh / oh)
            target['boxes'] = boxes
        return image, target

class RandomHorizontalFlip(object):
    """Random horizontal flip for PIL image and adjust boxes."""
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            w, h = image.size
            if 'boxes' in target and target['boxes'].numel() > 0:
                boxes = target['boxes'].clone()
                x_min = boxes[:, 0].clone()
                x_max = boxes[:, 2].clone()
                boxes[:, 0] = w - x_max
                boxes[:, 2] = w - x_min
                target['boxes'] = boxes
        return image, target
