import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm
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
import argparse
parser = argparse.ArgumentParser(description='Specify Parameters')

parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
parser.add_argument('model', metavar='model', type=str, choices=['unet', 'swin_unetr', 'utnet'], help='Specify a model')

parser.add_argument('norm_name', metavar='norm_name',  help='Specify a normalisation method')
# parser.add_argument('model_text', metavar='model_text', type=str, help='Describe your mode')
parser.add_argument('--base_c', metavar='--base_c', default = 12,type=int, help='base_channel which is the first output channel from first conv block')


args = parser.parse_args()
lr, batch_size, gpu_index, model_name, norm_name = args.lr, args.b_s, args.gpu_index, args.model, args.norm_name
base_c = args.base_c

print(torch.cuda.is_available())
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
def train_batch(images,labels,model,optimizer,criterion):
    images,labels = images.to(device,dtype=torch.float32),labels.to(device,dtype=torch.float32)
    outputs = model(images)
    loss = criterion(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
def train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

def save_model(path,name):
    artifact = wandb.Artifact(name=name, type="model")
    # Add the model file to the artifact
    artifact.add_file(path)
    # Log the artifact as an output of the run
    wandb.run.log_artifact(artifact)

def test(model,test_loader,criterion,config,best_valid_score):
    model.eval()

    with torch.no_grad():
        val_score = 0
        f1_score_record = np.zeros(4)
        total = 0
        for images,labels in test_loader:
            images,labels = images.to(device,dtype=torch.float32),labels.to(device)
            outputs = model(images).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = criterion(outputs,labels)
            val_score += score[1].item()/2 + score[2].item()/2
            f1_score_record += score
            total += labels.size(0)
        f1_score_record /= len(test_loader) #currently only works with batchsize=1
        val_score /= len(test_loader)
        print(f"model tested on {total} images" +
                  f"val_score: {val_score:%} f1_scores {f1_score_record}")
        wandb.log({"val_score": val_score, "Validation Background F1":f1_score_record[0],"Validation Outer Ring F1":f1_score_record[1],
                   "Validation Cup F1": f1_score_record[2],"Validation Disk F1": f1_score_record[3]})


            # Save the model in the exchangeable ONNX format
        torch.onnx.export(model, images, "model.onnx")
        wandb.save("model.onnx")

    model.train()

    if val_score > best_valid_score[0]:
        data_str = f"Valid score improved from {best_valid_score:2.8f} to {val_score:2.8f}. Saving checkpoint: {checkpoint_path_lowloss}"
        print(data_str)
        best_valid_score[0] = val_score
        torch.save(model.state_dict(), config.low_loss_path)
        save_model(config.low_loss_path,"low loss model")

    return val_score


def train(model, loader,test_loader, criterion, eval_criterion,optimizer, config):

    wandb.watch(model,criterion,log='all',log_freq=50) #this is freq of gradient recordings

    example_ct = 0
    batch_ct = 0

    best_valid_score = [0.0]#in list so I can alter it in test function
    for epoch in range(config.epochs):
        avg_epoch_loss = 0.0
        start_time = time.time()
        for _,(images, labels) in enumerate(loader):

            loss = train_batch(images,labels,model,optimizer,criterion)
            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct +=1

            if ((batch_ct+1)%50)==0:
                train_log(loss,example_ct,epoch)


        valid_score = test(model,test_loader,eval_criterion,config,best_valid_score)
        avg_epoch_loss/=example_ct
        end_time - time.time()
        iteration_mins,iteration_secs = train_time(start_time,end_time)
        data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
        data_str += f'\t Val Score: {valid_score:.8f}\n'
        print(data_str)
    torch.save(model.state_dict(),config.final_path)
    save_model(config.final_path,"final model")




def get_data(train):
    if train == True:
        x = sorted(glob("/home/kebl6872/Desktop/new_data/REFUGE2/train/image/*"))
        y = sorted(glob("/home/kebl6872/Desktop/new_data/REFUGE2/train/mask/*"))
    else:
        x = sorted(glob("/home/kebl6872/Desktop/new_data/REFUGE2/val/image/*"))
        y = sorted(glob("/home/kebl6872/Desktop/new_data/REFUGE2/val/mask/*"))

    return train_test_split(x,y)

def make_loader(dataset,batch_size):
    loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,)
    return loader
def make(config):

    train,test = get_data(train=True),get_data(train=False)
    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False,num_workers=4)
    criterion =  DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    eval_criterion = f1_valid_score
    model = UNet(3,config.classes,config.kernels[0],config.norm_name)
    optimizer = torch.optim.Adam(model.parameters(),lr = config.learning_rate)

    return model,train_loader,test_loader,criterion,eval_criterion,optimizer
def model_pipeline(hyperparameters):
    with wandb.init(project="REFUGE_UNet",config=hyperparameters):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion,optimizer = make(config)
        print(model)

        train(model,train_loader,test_loader,criterion,eval_criterion,optimizer,config)

    return model

if __name__ == "__main__":
    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')
    seeding(42)
    data_save_path = f'/home/kebl6872/Desktop/new_data/REFUGE2/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}/'
    create_dir(data_save_path + 'Checkpoint')
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)


    config = dict(epochs=50, classes=3, kernels=[base_c],norm_name=norm_name, batch_size=batch_size, learning_rate=lr, dataset="REFUGE",
                  architecture=model_name,low_loss_path=checkpoint_path_lowloss,final_path=checkpoint_path_final)

    model = model_pipeline(config)