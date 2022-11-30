
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        '''Normalise tensity in range [-1,-1]'''
        mask = (mask-127.5)/127.5
        '''change 0 => 1, 1 =>0'''
        mask_cup=np.where(mask > 0, 0, 1)
        mask_disc=np.where((mask == 1) | (mask ==0), 0, 1)
        mask = np.stack((mask_cup, mask_disc), axis=0)
        mask = mask.astype(np.float32)
        '''convert numpy array into tensor'''
        mask = torch.from_numpy(mask)


        return image, mask

    def __len__(self):
        return self.n_samples