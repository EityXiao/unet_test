# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import torch
import json 
import math
from torchvision.transforms import transforms

class TrainDataset(Dataset):
    def __init__(self, path, way = 1):
        dirlist = os.listdir(path)
        import torchvision.transforms as transforms
        self.transform_BZ = transforms.Normalize(
            mean=[0],# 取决于数据集
            std=[1]
        )

        self.data = []
        self.target = []
        self.pos = []
        self.way = way
        for i in dirlist:
            filename = os.path.join(path, i)
            im = Image.open(filename)
            crop_im = im.crop((0, 465, 1935, 2400))
            crop_im = crop_im.resize((290, 290))
            im = np.array(crop_im).transpose((2, 1, 0))[0,:,:]
            self.data.append(im)
        with open('target.json', 'r') as f:
            d = json.load(f)
            x = d['x']
            y = d['y']
            if way == 1:
                s_ ,e_ = 0, 150
            elif way == 2:
                s_ ,e_ = 150, 300
            else:
                s_ ,e_ = 300, 400
            for ind_1 in range(s_, e_):
                tmp = []
                tar_20 = np.ones((290, 290))
                tmp_pos = []
                for ind_2 in range(len(x[ind_1])):
                    tar_p = np.zeros((290, 290))
                    x_c = int(x[ind_1][ind_2] *0.15)
                    y_c = int((y[ind_1][ind_2] - 465) *0.15)
                    tmp_pos.append((x_c, y_c))
                    for ii in range(290):
                        for jj in range(290):
                            tar_p[ii,jj] = math.exp(-0.01*((ii-x_c)**2+(jj-y_c)**2))
                    tar_20 = tar_20 - tar_p
                    tmp.append(tar_p)
                tmp.append(tar_20)
                self.pos.append(tmp_pos)
                self.target.append(tmp)
    def __getitem__(self, index):
        im = self.data[index]
        ta = self.target[index]
        im = torch.unsqueeze(torch.tensor(im, dtype=torch.float), 0)
        im = self.transform_BZ(im)
        ta = torch.tensor(ta, dtype=torch.float)
        if self.way == 2 or self.way == 3:
            pos = self.pos[index]
            return im, pos
        return im, ta


    def __len__(self):
        return len(self.target)






