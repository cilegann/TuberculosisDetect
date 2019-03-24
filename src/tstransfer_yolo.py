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
import keras.backend.tensorflow_backend as KTF

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vis.visualization import visualize_saliency, visualize_cam,overlay
import platform
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndimage
from evaluate_tools import plot_confusion_matrix,evaluate
height=131
width=420
vector_length=173056
epoch=200
vali_split=0.3

host=platform.node()
mode=os.sys.argv[1]
if(host=='ican-1080ti'and mode=='train' and ('-both' in os.sys.argv)):
    gpu='both'
else:
    gpu='single'

if(gpu=='single'):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    batch_size=64

data='./data_origin_distribution_version'
train_mapping_file=os.path.join(data,'YOLO9000_x_y_mapping.csv')
vali_mapping_file=os.path.join(data,'YOLO9000_vali_x_y_mapping.csv')
mappings=[train_mapping_file,vali_mapping_file]

polluted_train_basedir=os.path.join(data,'polluted')
positive_train_basedir=os.path.join(data,'positive')
negative_train_basedir=os.path.join(data,'negative')
polluted_vali_basedir=os.path.join(data,'vali/polluted')
positive_vali_basedir=os.path.join(data,'vali/positive')
negative_vali_basedir=os.path.join(data,'vali/negative')
basedirs=[polluted_train_basedir,positive_train_basedir,negative_train_basedir,polluted_vali_basedir,positive_vali_basedir,negative_vali_basedir]


if(host=='cilegann-PC' or gpu=='single'):
    batch_size=32
else:
    batch_size=64

index = 0
vali_index = 0

train_x_file_list = []
train_x_file_list_backup=[]
train_x = []
train_y = []
train_y_backup=[]
num_of_0=0
num_of_1=0

vali_x_file_list = []
vali_x_file_list_backup=[]
vali_x=[]
vali_y = []
vali_y_backup=[]

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
                    if 'txt' in filename:
                        pathName=os.path.join(root,filename)
                        if( ('jpg' in pathName) or ('png' in pathName) ):
                            f.write(pathName+','+str(i)+'\n')

###################################################################################

def read_x_y_mapping(train_or_vali,shuffle):
    if(train_or_vali=='train'):
        global train_x_file_list_backup
        global train_y_backup
        file_list=[]
        y=[]
        mapping_file=train_mapping_file
    else:
        global vali_x_file_list_backup
        global vali_y_backup
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
        train_x_file_list_backup=file_list
        train_y_backup=np_utils.to_categorical(np.asarray(y),3)
    else:
        vali_x_file_list_backup=file_list
        vali_y_backup=np_utils.to_categorical(np.asarray(y),3)

###################################################################################

def parser(string):
    fn=string.split(",")[0]
    if "negative" in fn:
        y=0
    elif "positive" in fn:
        y=1
    elif "polluted" in fn:
        y=2
    vector=np.asarray( list(map(float,string.split(",")[1].split(" "))) )
    label=np_utils.to_categorical(y,3)
    return vector,label

def vec_reader(path):
    with open(path,'r') as f:
        line=f.readline()
    return parser(line)

###################################################################################

def load_all_valid(vali_x_file_list):
    global vali_x
    vali_x = np.zeros([len(vali_x_file_list), 173056])
    for i,f in enumerate(vali_x_file_list):
        vali_x[i],tmp= vec_reader(f)
    #TODO backup

###################################################################################

def generate_valid_from_train():
    global train_x_file_list
    global train_y
    global vali_x_file_list
    global vali_y
    vali_x_file_list = train_x_file_list[ :math.ceil(len(train_x_file_list)*vali_split) ]
    vali_y = train_y [ :math.ceil(len(train_x_file_list)*vali_split) ]
    train_x_file_list = train_x_file_list [math.floor(len(train_x_file_list)*vali_split):]
    train_y = train_y [math.floor(len(train_x_file_list)*vali_split):]

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

        output = np.zeros([batch_size, vector_length])
        for i in range(batch_size):
            output[i],tmp=vec_reader(file_list[i])

        yield output, label_list

###################################################################################

def get_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(vector_length,)))
    model.add(Dense(units=1024,kernel_initializer='random_uniform',activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=1024,kernel_initializer='random_uniform',activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=2,activation='softmax'))
    model.summary()
    return model


###################################################################################

def generate_new_data(stage):
    global train_x_file_list
    global train_x_file_list_backup
    global train_y
    global train_y_backup
    global vali_x_file_list
    global vali_x_file_list_backup
    global vali_y
    global vali_y_backup
    global vali_x
    global num_of_0
    global num_of_1
    train_x_file_list=[]
    train_y=[]
    vali_x_file_list=[]
    vali_y=[]
    vali_x=[]
    num_of_0=0
    num_of_1=0
    if(stage=='0_0'):
        print("Creating set for [ 1- Negative]/[ 0- Positive+Polluted]")
        #train y and train file list
        for i,y in enumerate(train_y_backup):
            if(np.argmax(y)==0):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(1)
                num_of_1+=1
            else:
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(0)
                num_of_0+=1
        #vali y and vali file list
        for i,y in enumerate(vali_y_backup):
            if(np.argmax(y)==0):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(1)
            else:
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(0)
        #np util
        train_y=np_utils.to_categorical(np.asarray(train_y),2)
        vali_y=np_utils.to_categorical(np.asarray(vali_y),2)
        #load all vali
        load_all_valid(vali_x_file_list)

    elif(stage=='0_1'):
        print("Creating set for [ 1- Positive]/[ 0- Polluted]")
        #train y and train file list
        for i,y in enumerate(train_y_backup):
            if(np.argmax(y)==1):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(1)
                num_of_1+=1
            elif(np.argmax(y)==2):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(0)
                num_of_0+=1
        #vali y and vali file list
        for i,y in enumerate(vali_y_backup):
            if(np.argmax(y)==1):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(1)
            elif(np.argmax(y)==2):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(0)
        #np util
        train_y=np_utils.to_categorical(np.asarray(train_y),2)
        vali_y=np_utils.to_categorical(np.asarray(vali_y),2)
        #load all vali
        load_all_valid(vali_x_file_list)

    elif(stage=='1_0'):
        print("Creating set for [ 1- Polluted]/[ 0- Negative+Positive]")
        #train y and train file list
        for i,y in enumerate(train_y_backup):
            if(np.argmax(y)==2):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(1)
                num_of_1+=1
            else:
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(0)
                num_of_0+=1
        #vali y and vali file list
        for i,y in enumerate(vali_y_backup):
            if(np.argmax(y)==2):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(1)
            else:
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(0)
        #np util
        train_y=np_utils.to_categorical(np.asarray(train_y),2)
        vali_y=np_utils.to_categorical(np.asarray(vali_y),2)
        #load all vali
        load_all_valid(vali_x_file_list)
 
    elif(stage=='1_1'):
        print("Creating set for [ 1- Negative]/[ 0- Positive]")
        #train y and train file list
        for i,y in enumerate(train_y_backup):
            if(np.argmax(y)==0):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(1)
                num_of_1+=1
            elif(np.argmax(y)==1):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(0)
                num_of_0+=1
        #vali y and vali file list
        for i,y in enumerate(vali_y_backup):
            if(np.argmax(y)==0):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(1)
            elif(np.argmax(y)==1):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(0)
        #np util
        train_y=np_utils.to_categorical(np.asarray(train_y),2)
        vali_y=np_utils.to_categorical(np.asarray(vali_y),2)
        #load all vali
        load_all_valid(vali_x_file_list)

    elif(stage=='2_0'):
        print("Creating set for [ 1- Positive]/[ 0- Negative+Polluted]")
        #train y and train file list
        for i,y in enumerate(train_y_backup):
            if(np.argmax(y)==1):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(1)
                num_of_1+=1
            else:
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(0)
                num_of_0+=0
        #vali y and vali file list
        for i,y in enumerate(vali_y_backup):
            if(np.argmax(y)==1):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(1)
            else:
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(0)
        #np util
        train_y=np_utils.to_categorical(np.asarray(train_y),2)
        vali_y=np_utils.to_categorical(np.asarray(vali_y),2)
        #load all vali
        load_all_valid(vali_x_file_list)

    elif(stage=='2_1'):
        print("Creating set for [ 1- Negative]/[ 0- Polluted]")
        #train y and train file list
        for i,y in enumerate(train_y_backup):
            if(np.argmax(y)==0):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(1)
                num_of_1+=1
            elif(np.argmax(y)==2):
                train_x_file_list.append(train_x_file_list_backup[i])
                train_y.append(0)
                num_of_0+=1
        #vali y and vali file list
        for i,y in enumerate(vali_y_backup):
            if(np.argmax(y)==0):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(1)
            elif(np.argmax(y)==2):
                vali_x_file_list.append(vali_x_file_list_backup[i])
                vali_y.append(0)
        #np util
        train_y=np_utils.to_categorical(np.asarray(train_y),2)
        vali_y=np_utils.to_categorical(np.asarray(vali_y),2)
        #load all vali
        load_all_valid(vali_x_file_list)
    else:
        raise(Exception("UNKNOWN stage : "+stage))
    return True

###################################################################################

def training(stage):
    if(generate_new_data(stage)==False):
        return
    model=get_model()
    es=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    mck=ModelCheckpoint(filepath=(stage+'_transferyolo_model_best.h5'),monitor='val_loss',save_best_only=True)
    if(host=='ican-1080ti' and gpu == 'both'):
        model = multi_gpu_model(model, gpus=2)
    class_weight = {0:1,1:num_of_0/num_of_1}
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit_generator(data_generator(True),validation_data=(vali_x,vali_y),validation_steps=1,steps_per_epoch=len(train_x_file_list)//batch_size, epochs=epoch,callbacks=[mck,es],class_weight=class_weight)
    model.save(stage+'_transferyolo_model.h5')
###################################################################################

def predict(form,best=True):
    global vali_x_file_list
    global vali_x
    global vali_y
    if(best):  
        model_0='./models/'+form+'_0_transferyolo_model_best.h5'
        model_1='./models/'+form+'_1_transferyolo_model_best.h5'
    else:
        model_0='./models/'+form+'_0_transferyolo_model.h5'
        model_1='./models/'+form+'_1_transferyolo_model.h5'
    print("=== Predict based on "+model_0+' and '+model_1+' ===')
    print("Loading "+model_0)
    model=load_model(model_0)
    print("Predicting "+model_0)
    prob_y_0=model.predict(vali_x)

    print("Loading "+model_1)
    model=load_model(model_1)
    print("Predicting "+model_1)
    prob_y_1=model.predict(vali_x)

    print("Merging result")
    result=np.zeros([len(vali_x_file_list),3])
    if(form=='0'):
        for i in range(len(vali_x_file_list)):
            result[i][0]=prob_y_0[i][1]
            result[i][1]=prob_y_0[i][0]*prob_y_1[i][1]
            result[i][2]=prob_y_0[i][0]*prob_y_1[i][0]
    elif(form=='1'):
        for i in range(len(vali_x_file_list)):
            result[i][0]=prob_y_0[i][0]*prob_y_1[i][1]
            result[i][1]=prob_y_0[i][0]*prob_y_1[i][0]
            result[i][2]=prob_y_0[i][1]
    elif(form=='2'):
        for i in range(len(vali_x_file_list)):
            result[i][0]=prob_y_0[i][0]*prob_y_1[i][1]
            result[i][1]=prob_y_0[i][1]
            result[i][2]=prob_y_0[i][0]*prob_y_1[i][0]
    with open('tsyolo_prob.csv','w') as file:
        file.write("filename,real_value,prob0,prob1,prob2\n")
        for f,r,p in zip(vali_x_file_list,vali_y,result):
            file.write(f+','+str(np.argmax(r)))
            for prob in p:
                file.write(","+str(prob))
            file.write("\n")
    y_true=np.argmax(vali_y,axis=1)
    y_pred=np.argmax(result,axis=1)
    plot_confusion_matrix(y_true,y_pred,["陰性","陽性","污染"])
    evaluate(y_true,y_pred)
    with open('./two_stage_transfer_result_'+form+'.csv','w') as file:
        file.write('file,true,pred,0_0,0_1,1_0,1_1\n')
        for i in range(len(vali_x_file_list)):
            file.write(vali_x_file_list[i]+','+str(y_true[i])+','+str(y_pred[i])+','+str(prob_y_0[i][0])+','+str(prob_y_0[i][1])+','+str(prob_y_1[i][0])+','+str(prob_y_1[i][1])+'\n')



    
###################################################################################

def main():
    if(mode=='train'):
        if('-s' in os.sys.argv):
            s=os.sys.argv[os.sys.argv.index('-s')+1]
        #sets=['0_1','1_0','1_1','2_0','2_1']
        print("\n   <<< Training mode >>>")
        read_x_y_mapping('train',True)
        read_x_y_mapping('vali',False)
        
        print("\n   <<< Training "+s+" >>>")
        training(s)
    elif(mode=='predict'):
        global vali_x_file_list
        global vali_y
        global vali_x
        read_x_y_mapping('vali',False)
        vali_x_file_list=vali_x_file_list_backup
        vali_y=vali_y_backup
        load_all_valid(vali_x_file_list)
        print(np.shape(vali_x))
        form=os.sys.argv[2]
        predict(form,True)
if __name__ == "__main__":
    main()
