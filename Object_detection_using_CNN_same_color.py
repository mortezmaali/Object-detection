# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 08:59:44 2021

@author: mamiri
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import scipy.io
import os
import cv2 as cv2
import pdb
import glob
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras import layers
from keras.layers import Conv2D, UpSampling2D, InputLayer, Dense
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
import matplotlib.pyplot as plt
import matplotlib
from tkinter import Tcl

# Create images with random rectangles and bounding boxes. 
num_imgs = 50000

img_size = 8
min_object_size = 1
max_object_size = 4
num_objects = 1

bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size,3))  # set background to 0

for i_img in range(num_imgs):
    for i_object in range(num_objects):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        imgs[i_img, x:x+w, y:y+h,:] = 1.  # set rectangle to 1
        bboxes[i_img, i_object] = [x, y, w, h]

# Reshape and normalize the image data to mean 0 and std 1. 
X = (imgs - np.mean(imgs)) / np.std(imgs)

# Normalize x, y, w, h by img_size, so that all values are between 0 and 1.
# Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
y = bboxes.reshape(num_imgs, -1) / img_size

# Split training and test.
num_imgs = int(0.8 * num_imgs)
train_X = X[0:num_imgs-2]
test_X = X[num_imgs-2:num_imgs]
train_y = y[0:num_imgs-2]
test_y = y[num_imgs-2:num_imgs]
test_imgs = imgs[num_imgs-2:num_imgs]
test_bboxes = bboxes[num_imgs-2:num_imgs]


# Build the model.
from keras.layers import  Activation, Dropout
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(InputLayer(input_shape=(8,8, 3)))
model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dense(4,activation='relu'))
model.compile(optimizer='Adadelta', loss='mse')
model.fit(train_X, train_y, epochs=450, verbose=2)

# Predict bounding boxes on the test images.
pred_y = model.predict(test_X)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


#plt.figure(figsize=(12, 3))
i=0
for i_subplot in range(1, len(test_X)+1):
    plt.subplot(1, 2, i_subplot)
    plt.imshow(test_imgs[i], cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i].reshape(-1, 4)):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[1], pred_bbox[0]), pred_bbox[3], pred_bbox[2], ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)), (pred_bbox[2], pred_bbox[3]+pred_bbox[1]+0.1), color='r')
    i=+1   


# Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset. 
summed_IOU = 0.
for pred_bbox, box_test in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
    summed_IOU += IOU(pred_bbox, box_test)
mean_IOU = summed_IOU / len(pred_bboxes)
print("IOU when using CNN is",mean_IOU)