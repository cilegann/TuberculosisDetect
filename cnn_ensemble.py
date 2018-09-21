import numpy as np
import os
import keras
from keras import utils as np_utils
from keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import random

host='1080ti' #1080ti / local
height=131
width=420

model_file_list=os.sys.argv[1:]
vali_mapping_file='./data/CNN_vali_x_y_mapping.csv'
model_folder='./cnn_model'
if (host=='local'):
    polluted_train_basedir='./original_data/categ/polluted'
    positive_train_basedir='./original_data/categ/positive'
    negative_train_basedir='./original_data/categ/negative'
    polluted_vali_basedir='./data/x'
    positive_vali_basedir='./data/p'
    negative_vali_basedir='./data/n'
if (host=='1080ti'):
    polluted_train_basedir='./data/polluted'
    positive_train_basedir='./data/positive'
    negative_train_basedir='./data/negative'
    polluted_vali_basedir='./data/vali/x'
    positive_vali_basedir='./data/vali/p'
    negative_vali_basedir='./data/vali/n'

vali_x_file_list = []
vali_x=[]
vali_y = []
prob_y=[]
def create_x_y_mapping():
    with open(vali_mapping_file,'w') as f:
        f.write("file_path,label\n")
        for root,directs,filenames in os.walk(positive_vali_basedir):
            for filename in filenames:
                pathName=os.path.join(root,filename)
                if( ('jpg' in pathName) or ('png' in pathName) ):
                    f.write(pathName+',1\n')
        for root,directs,filenames in os.walk(negative_vali_basedir):
            for filename in filenames:
                pathName=os.path.join(root,filename)
                if( ('jpg' in pathName) or ('png' in pathName) ):
                    f.write(pathName+',0\n')
        for root,directs,filenames in os.walk(polluted_vali_basedir):
            for filename in filenames:
                pathName=os.path.join(root,filename)
                if( ('jpg' in pathName) or ('png' in pathName) ):
                    f.write(pathName+',2\n')

def read_x_y_mapping():
    global vali_x_file_list
    global vali_y
    vali_x_file_list=[]
    vali_y=[]
    if( not os.path.exists(vali_mapping_file)):
        create_x_y_mapping()
    with open(vali_mapping_file,'r') as f:
        next(f)
        lines=f.readlines()
        for line in lines:
            vali_x_file_list.append(line.split(',')[0])
            vali_y.append(line.split(',')[1][:-1])
    vali_y = np.array(vali_y)
    vali_y = np_utils.to_categorical(vali_y,3)

def load_all_valid():
    global vali_x
    global prob_y
    vali_x = np.zeros([len(vali_x_file_list), height, width, 3])
    prob_y = np.zeros([len(vali_x_file_list),3])
    prob_y = prob_y.astype('float64')
    for i,f in enumerate(vali_x_file_list):
        vali_x[i]=Image.open(f).resize([width,height])
    vali_x=vali_x.astype('float64')
    vali_x/=255.

def resize_preprocessing(data,label):
    data=data.resize([width,height])
    data = np.asarray(data)
    data = data.astype('float64')
    
    if (random.random() > 0.5 and int(label[1])==1):
        data = cv2.flip(data, 1)
    data/=255.
    return data


read_x_y_mapping()
load_all_valid()
if(model_file_list[0] != 'all'):
    for m in model_file_list:
        model_name=os.path.join(model_folder,m)
        print("Predicting base on model: "+model_name )
        model=load_model(model_name)
        prob_y+=model.predict(vali_x)
else:
    for root,directs,filenames in os.walk(model_folder):
        for filename in filenames:
            model_name=os.path.join(root,filename)
            print("Predicting base on model: "+model_name)
            model=load_model(model_name)
            prob_y+=model.predict(vali_x)
print('Done')
with open('ensemble_result.csv','w') as file:
    file.write("filename,real_value,pred_value\n")
    for i,p in enumerate(vali_x_file_list):
        file.write(p+','+str(np.argmax(vali_y[i]))+','+str(np.argmax(prob_y[i]))+'\n')