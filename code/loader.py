'''
load train and test dataset
'''

import glob
import os

import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import numpy as np
import torch
import albumentations
from torch.utils.data import Dataset
# from det3d.datasets import build_dataloader, build_dataset
# from mmdet3d.datasets import build_dataloader, build_dataset
# from det3d.torchie import Config
# from torchpack.utils.config import configs
# from mmcv import Config
# from mmdet3d.utils import recursive_eval

# from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import cv2
import random


class Loaders:
    '''
    Initialize dataloaders
    '''

    def __init__(self, config):

        self.dataset_path = config.dataset_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size

        self.transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        train_set = CustomTrain()
        test_set = CustomTrain()

        self.train_loader = data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)


def take_threshold(data,threshold = (0.50, 0.45, 0.45, 0.40, 0.35, 0.40)): #(0.5,0.45,0.45,0.40,0.35,0.4)
    # data: 6, w, h
    # print("using nos threshold for C")
    for i in range(6):
        data[i] = data[i] > threshold[i]
    return data.astype(int)

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.pre_data = None
        self.gt_data = None
        self.flip = True
        # best threshold for the noisy map

        self.thre = (0.5,0.45,0.45,0.40,0.35,0.4)
        print("using threshold:", self.thre)
    def __len__(self):
        return len(self.gt_data)

    def __getitem__(self, i):
        res_example  = {}
        gt = np.load("dataset/train/" + self.gt_data[i], allow_pickle=True)
        # gt = np.load("dataset/train/" + self.gt_data[i]).astype(int)

        nos = np.load("dataset/train/" + self.pre_data[i]) 
        nos = take_threshold(nos,self.thre)
        
        if self.flip:
            if random.random()<0.5:
                gt = gt[:,::-1,:].copy()
                nos = nos[:,::-1,:].copy()
            if random.random()<0.5:
                gt = gt[:,:,::-1].copy()
                nos = nos[:,:,::-1].copy()

        res_example['image'] = gt
        res_example['nos'] = nos
        
        c,w,h = gt.shape # 6 200 200
        res_example['pure_nos'] = np.random.rand(1,w,h)
        # print(res_example['nos'].type)

        return res_example['image'], res_example['nos']

class CustomTrain(CustomBase):
    def __init__(self):
        super().__init__()

        # self.data = glob.glob("./data/train_"+training_images_list_file+"/gt_*.npy")
        self.gt_data = []
        self.pre_data = []
        imgs = os.listdir("dataset/train")
        for img in imgs:
            if img.startswith("gt_"):
                self.gt_data.append(img)
        for gt in self.gt_data:
            self.pre_data.append(gt.replace("gt","pre"))

        
        # for i in range(len(self.data)):
        #     self.gt_data.append("./data/train_"+training_images_list_file+"/gt_"+str(i)+".npy")
        #     self.pre_data.append("./data/train_"+training_images_list_file+"/pre_"+str(i)+".npy")

class CustomTest(CustomBase):
    def __init__(self):
        super().__init__()

        self.gt_data = []
        self.pre_data = []
        imgs = os.listdir("dataset/valid")
        for img in imgs:
            if img.startswith("gt_"):
                self.gt_data.append(img)
        for gt in self.gt_data:
            self.pre_data.append(gt.replace("gt","pre"))
        # self.data = glob.glob("./data/val_"+test_images_list_file+"/gt_*.npy")
        # self.flip = False
        # for i in range(len(self.data)):
        #     self.gt_data.append("./data/val_"+test_images_list_file+"/gt_"+str(i)+".npy")
        #     self.pre_data.append("./data/val_"+test_images_list_file+"/pre_"+str(i)+".npy")
