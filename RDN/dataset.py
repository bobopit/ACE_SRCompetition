# -*- coding: utf-8 -*-
# Author:
# date:
# functional description
"""

"""

import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import random
from random import randrange
from torchvision.transforms import Compose, CenterCrop, ToTensor


def input_transform():
    return Compose([
        #CenterCrop(crop_size),
        #Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def load_img_edge(filepath):  # single channel
    edge = Image.open(filepath).filter(ImageFilter.FIND_EDGES).convert('YCbCr').split()[0]
    # y, _, _ = img.split()
    return edge


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = torch.from_numpy(img_in.numpy()[:, :, ::-1].copy())
        img_tar = torch.from_numpy(img_tar.numpy()[:, :, ::-1].copy())
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = torch.from_numpy(img_in.numpy()[:, ::-1, :].copy())
            img_tar = torch.from_numpy(img_tar.numpy()[:, ::-1, :].copy())
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = torch.FloatTensor(np.transpose(img_in.numpy(), (0, 2, 1)))
            img_tar = torch.FloatTensor(np.transpose(img_tar.numpy(), (0, 2, 1)))
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


# class DatasetFromFolder(data.Dataset):
#     def __init__(self, image_dir, lr_dir, patch_size, upscale_factor, dataset, data_augmentation, input_transform=None, target_transform=None):
#         super(DatasetFromFolder, self).__init__()
#         self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
#         self.LRimage_filenames =[join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
#
#         self.lr_dir = lr_dir
#         self.patch_size = patch_size
#         self.upscale_factor = upscale_factor
#         self.dataset = dataset
#         self.input_transform = input_transform
#         self.target_transform = target_transform
#         self.data_augmentation = data_augmentation
#
#     def __getitem__(self, index):
#         target = load_img(self.image_filenames[index])
#         input = load_img(self.LRimage_filenames[index])
#         input_edge = load_img_edge(self.LRimage_filenames[index])
#         target_edge = load_img_edge(self.image_filenames[index])
#
#         # _, file = os.path.split(self.image_filenames[index])
#         #
#         # if self.dataset == 'DIV2K_train_LR_aug_x8':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x8.png'))
#         # elif self.dataset == 'DIV2K_train_LR_aug_x4':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))
#         # elif self.dataset == 'DIV2K_train_LR_aug_x2':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
#         # elif self.dataset == 'DIV2K_train_LR_difficult':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4d.png'))
#         # elif self.dataset == 'DIV2K_train_LR_mild':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4m.png'))
#         # elif self.dataset == 'DIV2K_train_LR_wild':
#         #     set=randrange(1, 5)
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4w'+str(set)+'.png'))
#
#         if self.input_transform:
#             input = self.input_transform(input)
#             input_edge = self.input_transform(input_edge)
#         if self.target_transform:
#             target = self.target_transform(target)
#             target_edge = self.target_transform(target_edge)
#         # input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor)
#
#         if self.data_augmentation:
#             input, target, _ = augment(input, target)
#         return input, target,input_edge,target_edge
#
#     def __len__(self):
#         return len(self.image_filenames)class DatasetFromFolder(data.Dataset):
#     def __init__(self, image_dir, lr_dir, patch_size, upscale_factor, dataset, data_augmentation, input_transform=None, target_transform=None):
#         super(DatasetFromFolder, self).__init__()
#         self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
#         self.LRimage_filenames =[join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
#
#         self.lr_dir = lr_dir
#         self.patch_size = patch_size
#         self.upscale_factor = upscale_factor
#         self.dataset = dataset
#         self.input_transform = input_transform
#         self.target_transform = target_transform
#         self.data_augmentation = data_augmentation
#
#     def __getitem__(self, index):
#         target = load_img(self.image_filenames[index])
#         input = load_img(self.LRimage_filenames[index])
#         input_edge = load_img_edge(self.LRimage_filenames[index])
#         target_edge = load_img_edge(self.image_filenames[index])
#
#         # _, file = os.path.split(self.image_filenames[index])
#         #
#         # if self.dataset == 'DIV2K_train_LR_aug_x8':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x8.png'))
#         # elif self.dataset == 'DIV2K_train_LR_aug_x4':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))
#         # elif self.dataset == 'DIV2K_train_LR_aug_x2':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
#         # elif self.dataset == 'DIV2K_train_LR_difficult':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4d.png'))
#         # elif self.dataset == 'DIV2K_train_LR_mild':
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4m.png'))
#         # elif self.dataset == 'DIV2K_train_LR_wild':
#         #     set=randrange(1, 5)
#         #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4w'+str(set)+'.png'))
#
#         if self.input_transform:
#             input = self.input_transform(input)
#             input_edge = self.input_transform(input_edge)
#         if self.target_transform:
#             target = self.target_transform(target)
#             target_edge = self.target_transform(target_edge)
#         # input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor)
#
#         if self.data_augmentation:
#             input, target, _ = augment(input, target)
#         return input, target,input_edge,target_edge
#
#     def __len__(self):
#         return len(self.image_filenames)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_root,folders, lr_dir, patch_size, upscale_factor, data_augmentation=False):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames=[]

        for folder in folders:
            HRImage_folders = os.listdir(os.path.join(image_dir_root,folder))
            HRImage_folders.sort()
            for image_folder in HRImage_folders:
                filenames=[join(image_dir_root,folder,image_folder,x) for x in listdir(os.path.join(image_dir_root,folder,image_folder)) if is_image_file(x)]
                self.image_filenames=self.image_filenames+filenames
        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        print(self.image_filenames[10001])
        self.LRimage_filenames=[]
        LRImage_folders=os.listdir(lr_dir)
        LRImage_folders.sort()
        for image_folder in LRImage_folders:
            filenames=[join(lr_dir,image_folder,x) for x in os.listdir(join(lr_dir,image_folder)) if is_image_file(x)]
            self.LRimage_filenames=self.LRimage_filenames+filenames
        # self.LRimage_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        print(self.LRimage_filenames[10001])
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.input_transform = input_transform()
        self.target_transform = target_transform()
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        input = load_img(self.LRimage_filenames[index])

        # _, file = os.path.split(self.image_filenames[index])
        #
        # if self.dataset == 'DIV2K_train_LR_aug_x8':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x8.png'))
        # elif self.dataset == 'DIV2K_train_LR_aug_x4':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))
        # elif self.dataset == 'DIV2K_train_LR_aug_x2':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
        # elif self.dataset == 'DIV2K_train_LR_difficult':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4d.png'))
        # elif self.dataset == 'DIV2K_train_LR_mild':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4m.png'))
        # elif self.dataset == 'DIV2K_train_LR_wild':
        #     set=randrange(1, 5)
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4w'+str(set)+'.png'))

        # if self.input_transform:
        #     input = self.input_transform(input)
        # if self.target_transform:
        #     target = self.target_transform(target)
        input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor)
        # input.save("./patch_image/input/{0:04d}.png".format(index))
        # target.save("./patch_image/target/{0:04d}.png".format(index))
        input=self.input_transform(input)
        target=self.target_transform(target)
        if self.data_augmentation:
            input, target, _ = augment(input, target)


        return input, target

    def __len__(self):
        return len(self.image_filenames)


# class DatasetFromFolderEval(data.Dataset):
#     def __init__(self, lr_dir,input_transform=None, target_transform=None):
#         super(DatasetFromFolderEval, self).__init__()
#         self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
#         self.input_transform = input_transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         input = load_img(self.image_filenames[index])
#         _, file = os.path.split(self.image_filenames[index])
#         input_edge = load_img_edge(self.image_filenames[index])
#         if self.input_transform:
#             input = self.input_transform(input)
#             input_edge = self.input_transform(input_edge)
#         return input, input_edge,file
#
#     def __len__(self):
#         return len(self.image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, lr_label_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.label_filenames = [join(lr_label_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        label = load_img(self.label_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
            label = self.input_transform(label)
        return input, label, file

    def __len__(self):
        return len(self.image_filenames)
