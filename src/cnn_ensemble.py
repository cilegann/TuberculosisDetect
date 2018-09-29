import numpy as np
import os
import keras
from keras import utils as np_utils
from keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import random

host='1080ti' #1080ti / local
height=131
width=420

model_file_list=os.sys.argv[1:]
vali_mapping_file='./data/CNN_vali_x_y_mapping.csv'
model_folder='./cnn_model'
if (host=='local'):
    polluted_vali_basedir='./data/x'
    positive_vali_basedir='./data/p'
    negative_vali_basedir='./data/n'
if (host=='1080ti'):
    polluted_vali_basedir='./data/vali/x'
    positive_vali_basedir='./data/vali/p'
    negative_vali_basedir='./data/vali/n'

vali_x_file_list = []
vali_x=[]
vali_y = []
prob_y=[]
def create_x_y_mapping():
    basedir_list=[negative_vali_basedir,positive_vali_basedir,polluted_vali_basedir]
    with open(vali_mapping_file,'w') as f:
        f.write("file_path,label\n")
        for i,b in enumerate(basedir_list):
            for root, directs,filenames in os.walk(b):
                for filename in filenames:
                    pathName=os.path.join(root,filename)
                    if( ('jpg' in pathName) or ('png' in pathName) ):
                        f.write(pathName+','+str(i)+'\n')

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
    data/=255.
    return data

def plot_confusion_matrix(cmx,classes,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cmx,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predict")
    plt.savefig('confusion_matrix.png')


read_x_y_mapping()
load_all_valid()
if(model_file_list[0] != 'all'):
    for m in model_file_list:
        model_name=os.path.join(model_folder,m)
        print("Predicting base on model: "+model_name )
        model=load_model(model_name)
        prob_y += model.predict(vali_x)
else:
    for root,directs,filenames in os.walk(model_folder):
        for filename in filenames:
            model_name=os.path.join(root,filename)
            print("Predicting base on model: "+model_name)
            model=load_model(model_name)
            prob_y +=model.predict(vali_x)

y=np.argmax(vali_y,axis=1)
pred_y=np.argmax(prob_y,axis=1)

for i,t in enumerate(y):
    print(str(i)+" "+str(t)+" -> "+str(pred_y[i]))

labels=["negative", "positive", "polluted"]
plt.figure()
cmx = confusion_matrix(y,pred_y)
cmx=cmx.astype('float')/cmx.sum(axis=1)[:,np.newaxis]
print(cmx)
plot_confusion_matrix(cmx,classes=labels,title='Confusion matrix')
plt.show()