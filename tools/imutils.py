import os
import os.path as osp
import random

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import voc12
import cv2

def save_img(path, img, cam=None):

    plt.imshow(img)
    if cam is not None:
        plt.imshow(cam, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()

class random_resize():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, sal=None, get_xy=False, xy=None, get_scale=False):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if xy:
            target_long = xy

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        if xy:
            img = img.resize(target_shape, resample=Image.NEAREST)
        else:
            img = img.resize(target_shape, resample=Image.CUBIC)
            
        if sal:
           sal = sal.resize(target_shape, resample=Image.CUBIC)
           return img, sal

        #################################################################################

        if get_scale:
            if w < h:
                scale = round(target_long/h)
            else:
                target_shape = (target_long, int(round(h * target_long / w)))

        #################################################################################

        if get_xy:
            return img, target_long
        else:
            return img


class random_crop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr, sal=None, get_xy=False, xy=None):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        if xy:
            cont_left, img_left, cont_top, img_top = xy

        if xy:
            container = 255*np.ones((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        else:
            container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = imgarr[img_top:img_top+ch, img_left:img_left+cw]
        if sal is not None:
            container_sal = np.zeros((self.cropsize, self.cropsize,1), np.float32)
            container_sal[cont_top:cont_top+ch, cont_left:cont_left+cw,0] = \
                sal[img_top:img_top+ch, img_left:img_left+cw]
            return container, container_sal

        if get_xy:
            xy = cont_left, img_left, cont_top, img_top
            return container, xy
        else:
            return container

class normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def denorm(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_denorm = transforms.Normalize(mean = [-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
                                     std = [1 / std_img[0], 1 / std_img[1], 1 / std_img[2]])

    return tf_denorm(img)

def norm_tensor(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_denorm = transforms.Normalize(mean = [mean_img[0], mean_img[1], mean_img[2]],
                                     std = [std_img[0], std_img[1], std_img[2]])

    return tf_denorm(img)


def HWC_to_CHW(tensor, sal=False):
    if sal:
        tensor = np.expand_dims(tensor, axis=0)
    else:
        tensor = np.transpose(tensor, (2, 0, 1))
    return tensor


def voc_palette(label):
	m = label.astype(np.uint8)
	r,c = m.shape
	cmap = np.zeros((r,c,3), dtype=np.uint8)
	cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
	cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
	cmap[:,:,2] = (m&4)<<5
	cmap[m==255] = [255,255,255]
	return cmap


def _crf_with_alpha(cam_dict, name, alpha=10):
        orig_img = np.ascontiguousarray(np.uint8(Image.open(os.path.join('./data/VOC2012/JPEGImages', name + '.jpg'))))
        v = np.array(list(cam_dict.values()))
        bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
        bgcam_score = np.concatenate((bg_score, v), axis=0)
        crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

        n_crf_al = dict()

        n_crf_al[0] = crf_score[0]
        for i, key in enumerate(cam_dict.keys()):
            n_crf_al[key + 1] = crf_score[i + 1]

        return n_crf_al

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))