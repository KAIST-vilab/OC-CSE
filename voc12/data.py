import os.path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import PIL.Image
from matplotlib import pyplot as plt
import tools.imutils as imutils

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def save_img(x, path):
    plt.imshow(x)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ImageSegDataset(Dataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, val_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.gt_path = gt_path

    def set_tf(self, phase):

        self.tf_rr = imutils.random_resize(256,768)
        self.tf_rc = imutils.random_crop(320)  

        self.tf_cj = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        self.tf_norm = imutils.normalize()

        self.tf_permute = imutils.HWC_to_CHW

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)
        
        if phase=='train':
            self.tf_list.append(imutils.torch.from_numpy)
            
        self.tf = transforms.Compose(self.tf_list)   

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx, val_flag=False):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        label = PIL.Image.open(self.gt_path + '/' + name + '.png')

        if not val_flag:

            img, xy = self.tf_rr(img, get_xy=True)
            label = self.tf_rr(label, xy=xy)

            if random.random()<0.2:
                transforms.functional.hflip(img)
                transforms.functional.hflip(label)

            img = self.tf_cj(img)

            img = np.asarray(img)
            label = np.expand_dims(np.asarray(label), axis=2)

            img = self.tf_norm(img)

            img, xy = self.tf_rc(img, get_xy=True)
            label = self.tf_rc(label, xy=xy)

            img = self.tf(img)
            label = self.tf(label)

        return name, img, label

class VOC12ImageSegDatasetMSF(VOC12ImageSegDataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(gt_path, img_name_list_path, voc12_root, val_flag=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def set_tf(self):

        self.tf_norm = imutils.normalize()
        self.tf_permute = imutils.HWC_to_CHW

    def __getitem__(self, idx):

        name, img, label = super().__getitem__(idx, val_flag=True)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        ms_label_list = []

        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            s_label = label.resize(target_size, resample=PIL.Image.NEAREST)
            
            ms_img_list.append(s_img)
            ms_label_list.append(s_label)

        for i in range(len(ms_img_list)):
                
            ms_img_list[i] = np.asarray(ms_img_list[i])
            ms_img_list[i] = self.tf_norm(ms_img_list[i])
            ms_img_list[i] = self.tf_permute(ms_img_list[i])
                
            ms_label_list[i] = np.expand_dims(np.asarray(ms_label_list[i]), axis=2)
            ms_label_list[i] = self.tf_permute(ms_label_list[i])

        msf_img_list = []
        msf_label_list = []
        
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_label_list.append(ms_label_list[i])
            msf_label_list.append(np.flip(ms_label_list[i], -1).copy())

        msf_img_list_t = []
        msf_label_list_t = []

        for i in range(len(msf_img_list)):
            msf_img_list_t.append(torch.from_numpy(msf_img_list[i]))
            msf_label_list_t.append(torch.from_numpy(msf_label_list[i].copy()))

        return name, msf_img_list_t, msf_label_list_t

