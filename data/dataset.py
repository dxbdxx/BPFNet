""" Palmprint datasets """
import os
import cv2
import math
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms

import scipy.io as scio
from matplotlib import pyplot as plt
import pandas as pd
from functools import cmp_to_key
from PIL import Image

import data.utils as utils


class CUHKSZ_align(torch.utils.data.Dataset):
    def __init__(self, data_path, split):
        self.img1_path = os.path.join(data_path, 'imgs')
        self.img2_path = os.path.join(data_path, 'imgs2')
        self.anno_path = os.path.join(data_path, 'marks')
        self.disp_path = os.path.join(data_path, 'disp')
        self.split = split
        self.img_list_ori = os.listdir(self.img1_path)

        def cmp(x,y):
            x0 = int(x[2:7])
            x1 = int(x.split('_')[-1][:-4])
            y0 = int(y[2:7])
            y1 = int(y.split('_')[-1][:-4])
            return x0*100+x1-y0*100-y1

        self.img_list_ori.sort(key=cmp_to_key(cmp))
        self.img1_list = []
        self.img2_list = []
        self.marks = []
        self.disps = []
        self.ID_list = []

        k = 0
        self.register_list =[]
        self.test_list = []
        for im in self.img_list_ori:
            im_str = im.split('_')
            subject = int(im_str[1])
            num = int(im_str[2].split('.')[0])
            if num > 6:
                hand = 'right'
                id = 2 * subject
            else:
                hand = 'left'
                id = 2 * subject + 1

            if split == 'train':
                if subject % 5 != 0:
                    self.img1_list.append(os.path.join(self.img1_path, im))
                    self.img2_list.append(os.path.join(self.img2_path, im))
                    f_mat_path = os.path.join(self.anno_path, im[:-3] + 'mat')
                    f_mat = scio.loadmat(f_mat_path)
                    self.disps.append(self._read_disp(os.path.join(self.disp_path, im[:-3] + 'txt')))
                    self.marks.append(f_mat['marks'])
                    self.ID_list.append(id)
            elif split == 'test':
                if subject % 5 == 0:
                    self.img1_list.append(os.path.join(self.img1_path, im))
                    self.img2_list.append(os.path.join(self.img2_path, im))
                    f_mat_path = os.path.join(self.anno_path, im[:-3] + 'mat')
                    f_mat = scio.loadmat(f_mat_path)
                    self.disps.append(self._read_disp(os.path.join(self.disp_path, im[:-3] + 'txt')))
                    self.marks.append(f_mat['marks'])
                    self.ID_list.append(id)
                    if num >= 4:
                        self.register_list.append(k)
                    else:
                        self.test_list.append(k)
                    k += 1

        label_unique = np.unique(np.array(self.ID_list))
        self.num_classes = len(label_unique)
        label_max = label_unique.max()
        self.class2label = np.zeros(label_max+1)
        for i in range(self.num_classes):
            self.class2label[label_unique[i]] = i

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img1_list)

    def _read_disp(self, path):
        f = open(path)
        line = f.readline()
        line_split = line.split()
        disp_w = int(line_split[0])
        disp_h = int(line_split[1])
        return [disp_w, disp_h]

    def _get_bbox_rot(self, marks):
        p1 = (marks[0] + marks[1]) / 2
        p2 = (marks[2] + marks[3]) / 2

        delta_12 = p1 - p2
        k = -delta_12[0] / delta_12[1]
        theta = np.arctan(k)
        midp = (p1 + p2) / 2

        ln = np.linalg.norm(delta_12)
        dp = 0.85 * ln
        x_center = dp * np.cos(theta) + midp[0]
        y_center = dp * np.sin(theta) + midp[1]

        l = ln * 1.25
        x1, y1 = x_center - l / 2, y_center - l / 2
        x3, y3 = x_center + l / 2, y_center + l / 2

        return np.array([x1, y1, x3, y3]), theta*180/np.pi

    def __getitem__(self, index):
        ## ----------- recognition --------------
        filename1 = self.img1_list[index]
        filename2 = self.img2_list[index]
        subject = int(self.class2label[self.ID_list[index]])

        img_ori1 = Image.open(filename1).convert('RGB')
        img_ori1_np = np.array(img_ori1)

        img_ori2 = Image.open(filename2).convert('RGB')
        img_ori2_np = np.array(img_ori2)

        img_ori = np.stack([img_ori1_np, img_ori2_np])
        img_ori = img_ori / 255
        img_ori = (img_ori - self.mean.reshape(1,1,1,3)) / self.std.reshape(1,1,1,3)
        img_ori_tensor = torch.Tensor(img_ori.transpose(0, 3, 1, 2))

        ## ----------- detection --------------
        bbox, theta = self._get_bbox_rot(self.marks[index])
        disp = self.disps[index]

        output_h = 48
        output_w = 64
        # output_h = 192
        # output_w = 256

        bbox = bbox/4
        bbox = bbox/4
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        hm = np.zeros((1, output_h, output_w), dtype=np.float32)

        radius = utils.gaussian_radius((math.ceil(w), math.ceil(h)))
        radius = max(0, int(radius))
        ct_int = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]).round()
        hm[0] = utils.draw_umich_gaussian(hm[0], ct_int, radius, theta)

        if self.split == 'train':
            return img_ori_tensor, subject, torch.Tensor(hm), torch.Tensor(disp), torch.Tensor([w, h]), torch.Tensor(bbox), theta.astype('float32')
        else:
            return img_ori_tensor, subject, torch.Tensor(disp)*4, torch.Tensor(bbox), theta.astype('float32')

