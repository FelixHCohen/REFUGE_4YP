import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd


basic_transform = A.Compose([A.VerticalFlip(p=0.0),A.HorizontalFlip(p=0.4),A.Rotate(limit=25, p=0.6,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0)
           ]) #prev probs were 0.15 0.45, 25 degree limit
class train_test_split(Dataset):
    def __init__(self, images_path, masks_path, disc_only=False,transform=False,return_path=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        self.return_path = return_path
        if transform:
            self.transform = basic_transform
        else:
            self.transform = False


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask < 128, 2, mask)  # set cup values to 2
        mask = np.where(mask == 128, 1, mask)  # disc pixels sset to 1
        mask = np.where(mask > 128, 0, mask)  # background pixels set to 0

        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """


        if self.disc_only:
            mask = np.where(mask == 2, 1, mask)  # set cup values to 2

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)
        if self.return_path == True:
            res = image, mask,self.images_path[index],self.masks_path[index]
        else:
            res = image,mask

        return res

    def __len__(self):
        return self.num_samples

class crop_dataset(train_test_split):

    def __init__(self,images_path,masks_path,disc_only=False,transform=False):
        super().__init__(images_path,masks_path,disc_only)

    def get_np_image(self,index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)

        return image

    def __getitem__(self,index):
        np_image = self.get_np_image(index)
        image,mask = super().__getitem__(index)

        return image,mask,np_image

class cup_dataset(train_test_split):
    def __init__(self, images_path, masks_path, disc_only=False,transform=False):
        super().__init__(images_path, masks_path, disc_only)


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        '''Normalise tensity in range [-1,-1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask < 128, 1, mask)  # set cup values to 1
        mask = np.where(mask >= 128, 0, mask)  # background pixels set to 0



        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)

        return image, mask


class GS1_dataset(Dataset):
    def __init__(self, images_path, cup_path,disc_path, disc_only=False,transform=False):

        self.images_path = images_path
        self.cup_path = cup_path
        self.disc_path = disc_path
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        if transform:
            self.transform = basic_transform
        else:
            self.transform = False


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        disc_mask = cv2.imread(self.disc_path[index], cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(self.cup_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(disc_mask >= 191, 1, disc_mask)  # set disc values to 1 w 75% agreement
        mask = np.where(disc_mask < 191, 0, mask)  # background to 0

        mask = np.where(cup_mask >= 191, 2, mask)  # set cup values to 2
        if self.disc_only:
            mask = np.where(mask>=2,1,mask)
        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """




        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)

        return image, mask

    def __len__(self):
        return self.num_samples


class RIMDL_dataset(GS1_dataset):

    def __init__(self,images_path,disc_path,cup_path,disc_only=False,transform=False):
        super().__init__(images_path,disc_path,cup_path,disc_only,transform)

    def __getitem__(self,index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        disc_mask = cv2.imread(self.disc_path[index], cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(self.cup_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(disc_mask >= 255, 1, disc_mask)  # set disc vals to 1
        mask = np.where(disc_mask < 255, 0, mask)  # background to 0

        if not self.disc_only:
            mask = np.where(cup_mask >= 255, 2, mask)  # set cup values to 2
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        """ Reading mask """

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)

        return image, mask

class small_prompt_dataset(Dataset):

    def __init__(self,df):
        super().__init__()

        self.image_paths = df['image_path']
        self.mask_paths = df['mask_path']
        self.i0 = df['i0']
        self.i1 = df['i1']
        self.i2 = df['i2']
        self.i3 = df['i3']
        self.j0 = df['j0']
        self.j1 = df['j1']
        self.j2 = df['j2']
        self.j3 = df['j3']
        self.v0 = df['v0']
        self.v1 = df['v1']
        self.v2 = df['v2']
        self.v3 = df['v3']

    def __getitem__(self,index):

        image = cv2.imread(self.image_paths[index][2:-3], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.mask_paths[index][2:-3], cv2.IMREAD_GRAYSCALE)

        mask = np.where(mask < 128, 2, mask)  # set cup values to 2
        mask = np.where(mask == 128, 1, mask)  # disc pixels sset to 1
        mask = np.where(mask > 128, 0, mask)  # background pixels set to 0
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)
        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)
        p0 = (self.i0[index],self.j0[index],self.v0[index])
        p1 = (self.i1[index], self.j1[index], self.v1[index])
        p2 = (self.i2[index], self.j2[index], self.v2[index])
        p3 = (self.i3[index], self.j2[index], self.v3[index])
        points = torch.tensor([p0,p1,p2,p3,])

        return image,mask,points
    def __len__(self):
        return len(self.image_paths)



