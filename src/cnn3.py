import os
import time
import tensorflow as tf
import numpy
import random
import numpy as np
import cv2
from PIL import Image
from keras import utils as np_utils
from keras.layers import *
from keras.models import Model,load_model,model_from_json
from keras import backend as K
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from Capsule_Keras import *
from evaluate_tools import plot_confusion_matrix,evaluate
import keras.backend.tensorflow_backend as KTF

width=420
height=131
num_of_classes=3
batch_size=32
train_mapping_file='./data/CNN_x_y_mapping.csv'
vali_mapping_file='./data/CNN_vali_x_y_mapping.csv'

polluted_train_basedir='./data/polluted'
positive_train_basedir='./data/positive'
negative_train_basedir='./data/negative'
polluted_vali_basedir='./data/vali/polluted'
positive_vali_basedir='./data/vali/positive'
negative_vali_basedir='./data/vali/negative'

def config_environment(args):
    global batch_size
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    batch_size=args.batch
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
                    if 'txt' not in filename:
                        pathName=os.path.join(root,filename)
                        if( ('jpg' in pathName) or ('png' in pathName) ):
                            f.write(pathName+','+str(i)+'\n')

###################################################################################

def read_x_y_mapping(train_or_vali,shuffle,args):
    file_list=[]
    y=[]
    if(train_or_vali=='train'):
        mapping_file=train_mapping_file
    else:
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
    else:
        s0=y.index('0')
        s1=y.index('1')
        s2=y.index('2')
        e0=s1-1
        e1=s2-1
        e2=len(y)-1
        l0=e0-s0+1
        l1=e1-s1+1
        l2=e2-s2+1
    return file_list,np_utils.to_categorical(np.array(y),num_of_classes)
    
###################################################################################

def preprocessing_augment(data,label):
    data=data.resize([width,height])
    data = np.asarray(data)
    data = data.astype('float64')
    if (random.random() > 0.5 and int(label[1])==1):
        data = cv2.flip(data, 1)
    data/=255.
    return data

###################################################################################

train_index=0
vali_index=0
train_indexes=[0,0,0]
def data_generator(is_training,file_lists,y,is_balanced):
    if is_balanced:
        pass
    else:
        global train_index
        global vali_index
        while(1):
            if is_training == True:
                if train_index + batch_size > len(file_lists):
                    train_index = 0
                train_index += batch_size
                index=train_index
            else:
                if vali_index + batch_size > len(file_lists):
                    vali_index = 0
                vali_index += batch_size
                index=vali_index
        
            file_list = file_lists[index-batch_size:index]
            label_list = y[index-batch_size:index]
            output = np.zeros([batch_size, height,width, 3])
            for i in range(batch_size):
                output[i]=preprocessing_augment(Image.open(file_list[i]),label_list[i])
            yield output, label_list

###################################################################################