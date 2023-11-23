import os
import random
import numpy as np
import torch
import cv2
import sys
import torch.nn as nn
from glob import glob
import matplotlib.pyplot as plt
from pode import Contour, Polygon, divide, Requirement,Point
from pode import joined_constrained_delaunay_triangles

def seeding(seed):  # seeding the randomness
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(file):
    if not os.path.exists(file):
        open(file, "w")
    else:
        print(f"{file} Exists")


def train_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def segmentation_score(y_true, y_pred, num_classes):
    # returns confusion matrix (TP, FP, TN, FN) for each class, plus a combined class for class 1+2 (disc)
    if y_true.size() != y_pred.size():
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros((num_classes + 1, 5))

    for i in range(num_classes):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        tn = np.sum(np.logical_and(y_true != i, y_pred != i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        accuracy = (tp + tn)/(tp+fp+tn+fn+smooth)
        precision = tp/(tp+fp+smooth)
        recall = tp/(tp+fn+smooth)
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        IoU = tp/(tp+fp+fn+smooth)
        score_matrix[i] = np.array([IoU, f1, recall, precision, accuracy])
    # DISC
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    IoU = tp / (tp + fp + fn + smooth)
    score_matrix[3] = np.array([IoU, f1, recall, precision, accuracy])

    return score_matrix

def f1_valid_two_classes(y_true,y_pred):
    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    # 0 == background class - will classify disk class as not 0 so I don't have to change the combined f1
    tp = np.sum(np.logical_and(y_true ==1, y_pred ==1))
    fp = np.sum(np.logical_and(y_true != 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true==1, y_pred==0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    return f1
def f1_valid_score(y_true, y_pred):
    if y_true.size() != y_pred.size():
        print(f' y true size: {y_true.size()} y_pred size: {y_pred.size()}')
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros(4)
    for i in range(3):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        score_matrix[i] = f1
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    score_matrix[3] = f1

    return score_matrix


def generate_point(y_true,y_pred,num):
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)

    combined_results = list()

    for class_idx in range(3):
        map = np.zeros((512,512))
        fn = (np.logical_and(y_true == class_idx, y_pred != class_idx))
        fn_indices = np.argwhere(fn == True)
        fn_indices = fn_indices[:,2:] # y_true indices are like [0,0,512,512]
        print(f'fn_indices: {fn_indices}')


        for i in range(fn_indices.shape[0]):
            map[fn_indices[i,0],fn_indices[i,1]]=1
        map = map.astype(np.uint8)
        plt.figure(figsize=(15, 15))
        plt.imshow(map,cmap='gray')
        plt.title(f"{class_idx}")

        (totalLabels, label_map, stats, centroids) = cv2.connectedComponentsWithStats(map, 8, cv2.CV_32S)
        centroids_w_stats = list()

        print(np.unique(label_map))
        np.set_printoptions(threshold=sys.maxsize)
        print('map')
        print(label_map[241,132])
        for a,b,c in zip(stats[1:], centroids[1:],list(range(1,totalLabels))):
            centroids_w_stats.append([a,b[::-1],c]) #reverse b as centroid returns x-y coordinates whilst we want indice of array

        for i in range(len(centroids_w_stats)):
            if y_true[0,0,round(centroids_w_stats[i][1][0]),round(centroids_w_stats[i][1][1])]!= class_idx:


                indices = np.argwhere(label_map==centroids_w_stats[i][2])
                print(f'indices for {centroids_w_stats[i][2]}: {indices[:10]}')
                l = list(range(indices.shape[0]))
                l_i = random.choice(l)
                centroids_w_stats[i][1] = indices[l_i,:]

        combined_results = combined_results + centroids_w_stats


    sorted_by_area = sorted(combined_results, key=lambda x: x[0][4])
    sorted_stats, sorted_centroids,sorted_labels = zip(*sorted_by_area)

    return [(round(x),round(y),y_true[0,0,round(x),round(y)]) for x,y in sorted_centroids[-1*num:]]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)                # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


def norm(input: torch.tensor, norm_name: str):
    if norm_name == 'layer':
        normaliza = nn.LayerNorm(list(input.shape)[1:])
    elif norm_name == 'batch':
        normaliza = nn.BatchNorm2d(list(input.shape)[1])
    elif norm_name == 'instance':
        normaliza = nn.InstanceNorm2d(list(input.shape)[1])

    normaliza = normaliza.to(f'cuda:{input.get_device()}')

    output = normaliza(input)

    return output


def get_lr(step, lr):
    if step <= 100:
        lr_ = lr
    if step > 100:
        lr_ = lr + lr * np.cos(2 * np.pi * step / 100)

    return lr_


def choose_test_set(test_data_num):
    test_x = 'nothing'
    if test_data_num == 0:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    elif test_data_num == 1:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_center_white_circle/*"))
    elif test_data_num == 2:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_corner_white_circle/*"))
    elif test_data_num == 3:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_edge_white_circle/*"))
    elif test_data_num == 4:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.1/*"))
    elif test_data_num == 5:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.2/*"))
    elif test_data_num == 6:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.3/*"))
    elif test_data_num == 7:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.4/*"))
    elif test_data_num == 8:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.5/*"))
    elif test_data_num == 9:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.1/*"))
    elif test_data_num == 10:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.2/*"))
    elif test_data_num == 11:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.3/*"))
    elif test_data_num == 12:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.4/*"))
    elif test_data_num == 13:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.5/*"))
    elif test_data_num == 14:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.1/*"))
    elif test_data_num == 15:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.2/*"))
    elif test_data_num == 16:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.3/*"))
    elif test_data_num == 17:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.4/*"))
    elif test_data_num == 18:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.5/*"))
    elif test_data_num == 19:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_match/*"))
    elif test_data_num == 20:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/greenlight/*"))
    return test_x
