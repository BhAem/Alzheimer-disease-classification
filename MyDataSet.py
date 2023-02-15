import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

class MyDataset(Dataset):
    def __init__(self, datas=None, labels=None, shape=None, input_D=None, input_H=None, input_W=None, phase='train', transforms=None):
        self.datas = datas
        self.labels = labels
        self.transforms = transforms
        self.shape = shape
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if self.phase == 'train':
            img = self.data_preprocess(self.datas[idx])
            label = 0
            if self.labels[idx] == 'AD':
                label = 0
            elif self.labels[idx] == 'MCI':
                label = 1
            elif self.labels[idx] == 'CN':
                label = 2
            img = torch.tensor(img)
            img = img.permute(3, 2, 0, 1)
            if self.transforms:
                img = self.transforms(img)
            return img,label
        elif self.phase == 'test':
            img = self.data_preprocess(self.datas[idx])
            img = torch.tensor(img)
            img = img.permute(3, 2, 0, 1)
            if self.transforms:
                img = self.transforms(img)
            return img

    def resize_data(self, data):
        if len(data.shape) == 3:
            [height, width, depth] = data.shape
            scale = [self.input_H * 1.0 / height, self.input_W * 1.0 / width, self.input_D * 1.0 / depth]
        else:
            [height, width, depth, channel] = data.shape
            scale = [self.input_H * 1.0 / height, self.input_W * 1.0 / width, self.input_D * 1.0 / depth, channel]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def data_preprocess(self, data):
        data = self.resize_data(data)
        cv2.normalize(data,data,0,255,cv2.NORM_MINMAX)
        if (len(data.shape) == 3):
            [x, y, z] = data.shape
        else:
            [x, y, z, _] = data.shape
        data = np.reshape(data, [x, y, z, 1])
        return data

