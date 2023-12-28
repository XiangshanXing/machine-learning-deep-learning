
import os
import cv2
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


def getDataPath(path):
    imgs_path = []
    masks_path = []
    data_path = path
    files = os.listdir(data_path)
    for f in files:
        if f.split('.')[0].split('_')[-1] != 'mask':
            imgs_path.append(os.path.join(data_path, f))
            masks_path.append(os.path.join(data_path, f.split('.')[0] + '_mask.png'))

    return imgs_path, masks_path


class ImageFold(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', transform_img=None, transform_mask=None):
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.imgs_path, self.masks_path = getDataPath(self.root)

    def __getitem__(self, idx):
        img_x = cv2.imread(self.imgs_path[idx])
        if len(np.shape(img_x)) != 3:
            img_x = cv2.cvtColor(img_x, cv2.COLOR_GRAY2RGB)
        else:
            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        output_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        if self.mode == 'train':
            mask_x = cv2.imread(self.masks_path[idx], cv2.IMREAD_GRAYSCALE)
            mask_x = cv2.resize(mask_x, (self.image_size, self.image_size), 0, 0, cv2.INTER_NEAREST)

            arterymask_crosscut = self.double_thresh(mask_x, 10, 70, 255)
            arterymask_slitting = self.double_thresh(mask_x, 100, 200, 255)
            # output_show = np.zeros((mask_x.shape[0], mask_x.shape[1]), dtype=np.uint8)
            # 横切
            inti = np.where(arterymask_crosscut > 200)
            if len(inti) != 0:
                for l1, l2 in zip(inti[0], inti[1]):
                    output_mask[l1][l2] = 1.0
                    # output_show[l1][l2] = 128

            # 纵切
            artery = np.where(arterymask_slitting > 200)
            if len(artery) != 0:
                for l1, l2 in zip(artery[0], artery[1]):
                    output_mask[l1][l2] = 2.0
                    # output_show[l1][l2] = 255

            # cv2.namedWindow('output_show', cv2.WINDOW_NORMAL)
            # cv2.imshow('output_show', output_show)
            # cv2.waitKey()
        if self.transform_img is not None:
            img_x = Image.fromarray(img_x.astype('uint8'))
            img_x = self.transform_img(img_x)

        output_mask = torch.from_numpy(output_mask).unsqueeze(0)
        # if self.transform_mask is not None:
        #     mask_x = self.transform_mask(output_mask)

        sample = {'image': img_x, 'label': output_mask, "path": self.imgs_path[idx]}
        return sample

    def __len__(self):
        return len(self.imgs_path)

    def double_thresh(self, img, tlow, thigh, maxval):
        if maxval != 0 and maxval != 255:
            return None
        img_mask = np.zeros(img.shape)
        img_mask = img_mask.fill(255 - maxval)
        if tlow > thigh:
            tlow, thigh = thigh, tlow
        t1, thresh_img1 = cv2.threshold(img, tlow, maxval, cv2.THRESH_BINARY)
        t2, thresh_img2 = cv2.threshold(img, thigh, maxval, cv2.THRESH_BINARY_INV)
        img_mask = cv2.bitwise_and(thresh_img1, thresh_img2)
        return img_mask
