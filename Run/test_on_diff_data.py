import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from tqdm import tqdm
import time
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_aug.data import train_test_split
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss
from utils import *
import cv2
import glob
from train_neat import get_data, get_gs1_or_rim_data
device = "cuda:0"



def make(config,path):
    model = UNet(3, config["classes"], config["base_c"], config["kernels"], config["norm_name"])
    if not os.path.exists(path):
        print("path does not exist")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.eval()
    train= get_data(train=True)
    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=False, )

    criterion = f1_valid_score
    return model,train_loader,criterion


def plot_output(output,image,label,score,point_tuples):
    color_map = {0:'red',1:'green',2:'blue'}
    image_np = image[0,:,:,:].cpu().numpy().transpose(1,2,0)
    image_np = ((image_np * 127.5)+127.5).astype(np.uint16)
    output = output[0,:,:,:].cpu().numpy().transpose(1,2,0)
    label = label[0,:,:,:].cpu().numpy().transpose(1,2,0)
    label = np.repeat(label,3,2)
    output = np.repeat(output,3,2)
    d = {0:0,1:128,2:255}
    vfunc = np.vectorize(lambda x: d[x])

    # Apply the vectorized function to the array
    label = vfunc(label)
    output = vfunc(output)


    rows = 1
    cols = 3

    # Create a figure with the specified size
    plt.figure(figsize=(15, 15))
    plt.subplot(rows, cols, 1)
    plt.imshow(image_np)
    plt.title("image")
    plt.axis('off')
    # Plot the mask in the even-numbered subplot
    plt.subplot(rows, cols, 2)
    plt.imshow(output)
    plt.title(f"output avg score: {score}")
    for y, x, val in point_tuples: #point tuples stored in index e.g. ij = y,x
        print(f'({x},{y}): {val}')
        circle = plt.Circle((x,y),3,color=color_map[val])
        plt.gca().add_patch(circle)
    plt.axis('off')
    plt.subplot(rows, cols, 3)
    plt.imshow(label,)
    plt.title("ground truth")
    plt.axis('off')



    # Show the plot
    plt.show()
def test_pipeline(config,model_path):
    model,loader,criterion = make(config,model_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        val_score = 0
        test_set_val = 0
        total = 0
        for i,(images,labels) in enumerate(loader):
            images,labels = images.to(device,dtype=torch.float32),labels.to(device)
            outputs = model(images).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = criterion(outputs,labels)

            val_score += score
            total += labels.size(0)

            if i < 10:
                point_tuples = generate_point(labels,outputs,4)


                plot_output(outputs,images,labels,score[1],point_tuples)



        val_score /= total #currently needs batch size 1

        print(f"cup F1 {val_score[2]}\ndisc F1: {val_score[3]}")

        return val_score[2],val_score[3]

seeding(42)
config = dict(classes=3,base_c=12,kernels=[6,12,24,48],norm_name='batch',batch_size=16,lr=3e-4,seed=147,dataset="")
print(config)
seeding(config["seed"])
model_paths = glob.glob(f'/home/kebl6872/Desktop/new_data/REFUGE2/test/1600_unet_{config["norm_name"]}_lr_{config["lr"]}_bs_{config["batch_size"]}_fs_{config["base_c"]}_[[]{"_".join(str(k) for k in config["kernels"])}[]]/Checkpoint/seed/**/*lowloss.pth',recursive=True)

df = pd.read_csv('/home/kebl6872/Desktop/new_data/testing_csvs/test1.csv') # columns: |dataset|seed|cup f1|disc f1
for model_path in model_paths:
    df_row = {'dataset':[],'seed':[],'cup f1':[],'disc f1':[]}
    seed = model_path.split("/")[10]
    print(f'seed: {seed}')

    df_row['dataset'].append(config["dataset"])
    df_row['seed'].append(seed)
    c_f1,d_f1 = test_pipeline(config,model_path)
    df_row['cup f1'].append(c_f1)
    df_row['disc f1'].append(d_f1)
    df = pd.concat([df,pd.DataFrame(df_row)],ignore_index=True)

df.to_csv('/home/kebl6872/Desktop/new_data/testing_csvs/test1.csv')