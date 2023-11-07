from glob import glob
import os
import cv2
import numpy as np
import torch
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)
ori_data_path = '/home/kebl6872/Desktop/new_data/REFUGE2/train/mask/g0005_3.bmp'

mask = cv2.imread(ori_data_path,cv2.IMREAD_GRAYSCALE)

cv2.imshow('img',mask)
mask = np.where(mask < 128, 2, mask)     # cup
mask = np.where(mask == 128, 1, mask)    # disc - cup = outer ring
mask = np.where(mask > 128, 0, mask)     # background
mask = mask.astype(np.int64)
mask = np.expand_dims(mask, axis=0)
      # (1,512,512)
