# coding:utf-8

import tensorflow as tf
import os
import random
import math
import shutil

import keras
import keras.backend as K
from keras import utils as np_utils
from keras import Sequential
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.layers import Activation,Dense,Dropout,MaxPooling2D,Flatten,Conv2D
from keras.optimizers import rmsprop,adam
import keras.losses
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vis.visualization import visualize_saliency, visualize_cam,overlay
import platform
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndimage

height=131
width=420
epoch=200
vali_split=0.3

host=platform.node()
mode=os.sys.argv[1]
if(host=='ican-1080ti'and mode=='train'):
    gpu=os.sys.argv[2]
else:
    gpu='single'

model_to_load=''
if(mode=='predict' or mode=='cam'):
    model_to_load=os.sys.argv[2]

if(gpu=='single'):
    os.environ['CUDA_VISIBLE_DEVICES']='0'

train_mapping_file='./data/CNN_x_y_mapping.csv'
vali_mapping_file='./data/CNN_vali_x_y_mapping.csv'

if (host=='cilegann-PC'):
    polluted_train_basedir='./original_data/categ/polluted'
    positive_train_basedir='./original_data/categ/positive'
    negative_train_basedir='./original_data/categ/negative'
    polluted_vali_basedir='./data/x'
    positive_vali_basedir='./data/p'
    negative_vali_basedir='./data/n'
if (host=='ican-1080ti'):
    polluted_train_basedir='./data/polluted'
    positive_train_basedir='./data/positive'
    negative_train_basedir='./data/negative'
    polluted_vali_basedir='./data/vali/x'
    positive_vali_basedir='./data/vali/p'
    negative_vali_basedir='./data/vali/n'

if(host=='cilegann-PC' or gpu=='single'):
    batch_size=32
else:
    batch_size=64

index = 0
vali_index = 0

train_x_file_list = []
train_x = []
train_y = []

vali_x_file_list = []
vali_x=[]
vali_y = []
prob_y=[]

###################################################################################

def create_x_y_mapping(train_or_vali):
    basedir_list=[]
    if(train_or_vali=='train'):
        mapping_file=train_mapping_file
        basedir_list=[negative_train_basedir,positive_train_basedir,polluted_train_basedir]
    else:
        mapping_file=vali_mapping_file
        basedir_list=[negative_vali_basedir,positive_vali_basedir,polluted_vali_basedir]
    with open(mapping_file,'w') as f:
        f.write("file_path,label\n")
        for i,b in enumerate(basedir_list):
            for root, directs,filenames in os.walk(b):
                for filename in filenames:
                    pathName=os.path.join(root,filename)
                    if( ('jpg' in pathName) or ('png' in pathName) ):
                        f.write(pathName+','+str(i)+'\n')

###################################################################################

def read_x_y_mapping(train_or_vali,shuffle):
    if(train_or_vali=='train'):
        global train_x_file_list
        global train_y
        file_list=[]
        y=[]
        mapping_file=train_mapping_file
    else:
        global vali_x_file_list
        global vali_y
        file_list=[]
        y=[]
        mapping_file=vali_mapping_file
    if(not os.path.exists(mapping_file)):
        create_x_y_mapping(train_or_vali)
    with open(mapping_file,'r') as f:
        next(f)
        lines=f.readlines()
        for line in lines:
            file_list.append(line.split(',')[0])
            y.append(line.split(',')[1][:-1])
    if(shuffle):
        c=list(zip(file_list,y))
        random.shuffle(c)
        file_list,y=zip(*c)
    if(train_or_vali=='train'):
        train_x_file_list=file_list
        train_y=np_utils.to_categorical(np.array(y),3)
    else:
        vali_x_file_list=file_list
        vali_y=np_utils.to_categorical(np.array(y),3)

###################################################################################

def load_all_valid():
    global vali_x
    vali_x = np.zeros([len(vali_x_file_list), height, width, 3])
    for i,f in enumerate(vali_x_file_list):
        vali_x[i]=Image.open(f).resize([width,height])
    vali_x=vali_x.astype('float64')
    vali_x/=255.

###################################################################################

def resize_preprocessing(data,label):
    data=data.resize([width,height])
    data = np.asarray(data)
    data = data.astype('float64')
    if (random.random() > 0.5 and int(label[1])==1):
        data = cv2.flip(data, 1)
    data/=255.
    return data

###################################################################################

def data_generator(is_training):
    global index
    global vali_index
    while(1):
        if is_training == True:
            if index + batch_size > len(train_x_file_list):
                index = 0
            file_list = train_x_file_list[index:index + batch_size]
            label_list = train_y[index:index + batch_size]
            index += batch_size
        else:
            if vali_index + batch_size > len(vali_x_file_list):
                vali_index = 0
            file_list = vali_x_file_list[vali_index:vali_index + batch_size]
            label_list = vali_y[vali_index:vali_index + batch_size]
            vali_index += batch_size

        output = np.zeros([batch_size, height,width, 3])
        for i in range(batch_size):
            output[i]=resize_preprocessing(Image.open(file_list[i]),label_list[i])

        yield output, label_list

###################################################################################

def get_model():
    model = Sequential()

    model.add(Conv2D(32,(3,3),strides=(1,1),input_shape=(height,width,3),data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),strides=(1,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()
    return model

###################################################################################

def training(stage):
    #TODO data of first stage
    #TODO first stage of training 
    #TODO data of second stage
    #TODO second stage of training
###################################################################################

def predict();
    #TODO: two stage of prediction

###################################################################################

def main():
    if(mode=='train')
if __name__ == "__main__":
    main()
