import torch
from torch.utils.data import Dataset

import itertools
import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy

from hawp.fsl.raw import RawSynthesisConfig, RawSynthesizer



def add_shade(img, random_state=None, nb_ellipses=20,
              amplitude=[-0.5, 0.5], kernel_size_interval=(250, 350)):
    """ Overlay the image with several shades
    Parameters:
      nb_ellipses: number of shades
      amplitude: tuple containing the illumination bound (between -1 and 0) and the
        shawdow bound (between 0 and 1)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    transparency = random_state.uniform(*amplitude)

    min_dim = min(img.shape[:2]) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        angle = random_state.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float64), (kernel_size, kernel_size), 0)
    shade = 1 - transparency * mask / 255.
    if img.ndim == 3:
        shade = shade[..., None]
    shaded = img * shade
    shaded = np.clip(shaded, 0, 255)
    return shaded.astype(np.uint8)


def add_fog(img, random_state=None, max_nb_ellipses=20,
            transparency=0.6, kernel_size_interval=(150, 250)):
    """ Overlay the image with several shades
    Parameters:
      max_nb_ellipses: number max of shades
      transparency: level of transparency of the shades (1 = no shade)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    centers = np.empty((0, 2), dtype=np.int32)
    rads = np.empty((0, 1), dtype=np.int32)
    min_dim = min(img.shape[:2]) / 4
    shaded_img = img.copy()
    for i in range(max_nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = random_state.randint(256)  # color of the shade
        angle = random_state.rand() * 90
        cv2.ellipse(shaded_img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    shaded_img = shaded_img.astype(float)
    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1

    cv2.GaussianBlur(shaded_img, (kernel_size, kernel_size), 0, shaded_img)
    mask = np.where(shaded_img != img)
    shaded_img[mask] = (1 - transparency) * shaded_img[mask] + transparency * img[mask]
    shaded_img = np.clip(shaded_img, 0, 255)
    return shaded_img.astype(np.uint8)


# def motion_blur(img, max_ksize=5):
def motion_blur(img, max_ksize=8):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_ksize+1)/2)*2 + 1  # make sure is odd

    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img.astype(np.uint8), -1, kernel)
    return img

# def random_brightness(img, random_state=None, max_change=50):
def random_brightness(img, random_state=None, max_change=80):
    """ Change the brightness of img
    Parameters:
      max_change: max amount of brightness added/subtracted to the image
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    brightness = random_state.randint(-max_change, max_change)
    new_img = img.astype(np.int16) + brightness
    return np.clip(new_img, 0, 255).astype(np.uint8)


# def random_contrast(img, random_state=None, max_change=[0.5, 1.5]):
def random_contrast(img, random_state=None, max_change=[0.5, 2.0]):
    """ Change the contrast of img
    Parameters:
      max_change: the change in contrast will be between 1-max_change and 1+max_change
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    contrast = random_state.uniform(*max_change)
    mean = np.mean(img, axis=(0, 1))
    new_img = np.clip(mean + (img - mean) * contrast, 0, 255)
    return new_img.astype(np.uint8)



class TrainDataset(Dataset):
    def __init__(
        self,
        root,
        ann_file,
        transform=None,
        augmentation=4,
        raw_config=None,
        raw_synthesizer=None,
        return_rgb=False,
        step_counter=None,
    ):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
        self.augmentation = augmentation
        self.return_rgb = return_rgb
        self.raw_config = self._normalize_raw_config(raw_config)
        if not self.return_rgb and raw_synthesizer is None and self.raw_config is None:
            raise ValueError("RawPLNet training requires DATASETS.RAW config; RGB fallback is disabled")
        self.raw_synthesizer = raw_synthesizer
        if not self.return_rgb and self.raw_synthesizer is None:
            self.raw_synthesizer = RawSynthesizer(
                self.raw_config,
                device=getattr(raw_config, "DEVICE", "cuda"),
            )
        self.step_counter = self._normalize_step_counter(step_counter)

    def _normalize_raw_config(self, raw_config):
        if raw_config is None or isinstance(raw_config, RawSynthesisConfig):
            return raw_config
        return RawSynthesisConfig.from_cfg(raw_config)

    def _normalize_step_counter(self, step_counter):
        if step_counter is None:
            return itertools.count()
        if isinstance(step_counter, int):
            return itertools.count(step_counter)
        return step_counter

    def _next_raw_step(self):
        if callable(self.step_counter):
            return int(self.step_counter())
        return int(next(self.step_counter))
    
    def __getitem__(self, idx_):
        # print(idx_)

        idx = idx_%len(self.annotations)
        # random_prob = torch.rand(1)
        # reminder = torch.randint(0,4,(1,)).item()
        reminder = idx_//len(self.annotations)
        
        # idx = 0
        # reminder = 0
        ann = copy.deepcopy(self.annotations[idx])
        if len(ann['edges_negative']) == 0:
            ann['edges_negative'] = [[0,0]]
        ann['reminder'] = reminder
        image_path = osp.join(self.root,ann['filename'])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_aug_random = random.randint(1, 7)
        if data_aug_random == 1:
            image = add_shade(image)
        elif data_aug_random == 2:
            image = add_fog(image)
        elif data_aug_random == 3:
            image = motion_blur(image)
        elif data_aug_random == 4:
            image = random_brightness(image)
        elif data_aug_random == 5:
            image = random_contrast(image)
        image = image[:,:,:3]

        # if len(ann['junctions']) == 0:
        #     ann['junctions'] = [[0,0]]
        #     ann['edges_positive'] = [[0,0]]
        
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        for key,_type in (['junctions',np.float32],
                          ['edges_positive',np.int32],
                          ['edges_negative',np.int32]):
            ann[key] = np.array(ann[key],dtype=_type)
        
        width = ann['width']
        height = ann['height']
        if reminder == 1:
            image = image[:,::-1,:]
            # image = F.hflip(image)
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
        elif reminder == 2:
            # image = F.vflip(image)
            image = image[::-1,:,:]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        elif reminder == 3:
            # image = F.vflip(F.hflip(image))
            image = image[::-1,::-1,:]
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        elif reminder == 4:
            image_rotated = np.rot90(image)

            junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
            theta = 0.5*np.pi
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            junctions_rotated = (rot_mat@junctions.transpose()).transpose()
            junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

            ann['width'] = image_rotated.shape[1]
            ann['height'] = image_rotated.shape[0]
            ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
            image = image_rotated
        elif reminder == 5:
            image_rotated = np.rot90(np.rot90(np.rot90(image)))

            junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
            theta = 1.5*np.pi
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            junctions_rotated = (rot_mat@junctions.transpose()).transpose()
            junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

            ann['width'] = image_rotated.shape[1]
            ann['height'] = image_rotated.shape[0]
            ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
            image = image_rotated
        # elif reminder == 6:

        if not self.return_rgb:
            image = self.raw_synthesizer.synthesize_rgb(image, iter_idx=self._next_raw_step())
            image = np.clip(image, 0.0, 1.0) * 255.0

        image = image.astype(float)

        if self.transform is not None:
            return self.transform(image,ann)
        
        
        return image, ann

    def __len__(self):
        return len(self.annotations)*self.augmentation

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
