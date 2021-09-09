import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from matplotlib.pyplot import Rectangle
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models

images_path = Path('D:/projects_python/road_signs_network/road_signs/images')
anno_path = Path('./road_signs/annotations')


def file_list(root, file_type):
    """returns a fully qualified list of file names under the root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if
            f.endswith(file_type)]


def generate_train_df(anno_path):
    annotations = file_list(anno_path, 'xml')
    anno_list = []

    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['filename'] = Path(str(images_path) + '/' + root.find('./filename').text)
        anno['width'] = root.find('./size/width').text
        anno['height'] = root.find('./size/height').text
        anno['class'] = root.find('./object/name').text
        anno['xmin'] = int(root.find('./object/bndbox/xmin').text)
        anno['ymin'] = int(root.find('./object/bndbox/ymin').text)
        anno['xmax'] = int(root.find('./object/bndbox/xmax').text)
        anno['ymax'] = int(root.find('./object/bndbox/ymax').text)
        anno_list.append(anno)

    return pd.DataFrame(anno_list)


df_train = generate_train_df(anno_path)

class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

print(df_train.shape)
print(df_train.head())


def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def create_mask(bb, x):
    """creates a mask for the bounding box with the same shape as the image"""
    rows, cols, *_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y


def mask_to_bb(Y):
    """converts mask Y to a bounding box, assuming 0 as background non-zero object"""
    cols, rows = np.nonzero(Y)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    bottom_row = np.max(rows)
    left_col = np.min(cols)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


def create_bb_array(x):
    """generates bounding box array from a df_train row"""
    return np.array([x[5], x[4], x[7], x[6]])


def resize_image_bb(read_path, write_path, bb, sz):
    """resizes an image and its bounding box and writes image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49 * sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49 * sz), sz))
    new_path = str(write_path / read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)


new_paths = []
new_bbs = []
train_path_resized = Path('./road_signs/images_resized')
for index, row in df_train.iterrows():
    new_path, new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values), 300)
    new_paths.append(new_path)
    new_bbs.append(new_bb)
df_train['new_path'] = new_paths
df_train['new_bb'] = new_bbs

im = cv2.imread(str(df_train.values[58][0]))
bb = create_bb_array(df_train.values[58])

Y = create_mask(bb, im)
mask_to_bb(Y)

plt.imshow(im)
plt.show()

plt.imshow(Y, cmap='gray')
plt.show()


def crop(im, r, c, target_r, target_c):
    return im[r:r + target_r, c:c + target_c]


def random_crop(x, r_pix=8):
    """returns a random crop"""
    r, c, *_ = x.shape
    c_pix = round(r_pix * c / r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2 * rand_r * r_pix).astype(int)
    start_c = np.floor(2 * rand_c * c_pix).astype(int)
    return crop(x, start_r, start_c, r - 2 * r_pix, c - 2 * c_pix)


def center_crop(x, r_pix=8):
    r, c, *_ = x.shape
    c_pix = round(r_pix * c / r)
    return crop(x, r_pix, c_pix, r - 2 * r_pix, c - 2 * c_pix)


def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """rotates an image by deg degrees"""
    r, c, *_ = im.shape
    M = cv2.getRotationMatrix2D((c / 2, r / 2), deg, 1)
    if y:
        return cv2.warpAffine(im, M, (c, r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS + interpolation)


def random_cropXY(x, Y, r_pix=8):
    """returns a random crop"""
    r, c, *_ = x.shape
    c_pix = round(r_pix * c / r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2 * rand_r * r_pix).astype(int)
    start_c = np.floor(2 * rand_c * c_pix).astype(int)
    xx = crop(x, start_r, start_c, r - 2 * r_pix, c - 2 * c_pix)
    YY = crop(Y, start_r, start_c, r - 2 * r_pix, c - 2 * c_pix)
    return xx, YY


def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random() - .50) * 20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5:
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color,
                         fill=False, lw=3)


def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))


# original
print(str(df_train.values[68][8]))
im = cv2.imread(str(df_train.values[68][8]))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[68][9])
