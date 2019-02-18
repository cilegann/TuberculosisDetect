
# coding: utf-8


import tensorflow as tf
import os
import keras
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
import random
import math
import matplotlib.pyplot as plt
import cv2
import platform
import shutil
from evaluate_tools import cam,plot_confusion_matrix,evaluate
#train : python3 scriptname train [single/both]
#predict: python3 scriptname predict [modelname]
#saliencymap: python3 scriptname saliencymap [modelname] [dataset] [portion] [amount] [save/show]

host = platform.node()  #cilegann-PC / ican-1080ti
mode = os.sys.argv[1] #train / predict / saliencymap
gpu='single'

if(mode=='predict' or mode=='cam'):
    model_to_load=os.sys.argv[2]
if(host=='ican-1080ti'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    batch_size=64
    if('balance' in os.sys.argv):
        batch_size=63

elif(host=='cilegann-PC'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    batch_size=32
    if('balance' in os.sys.argv):
        batch_size=30

positive_weigt=15.
polluted_weight=4.5
negative_weight=1.4
height=131
width=420
train_mapping_file='./data/CNN_x_y_mapping.csv'
vali_mapping_file='./data/CNN_vali_x_y_mapping.csv'
polluted_train_basedir='./data/polluted'
positive_train_basedir='./data/positive'
negative_train_basedir='./data/negative'
polluted_vali_basedir='./data/vali/polluted'
positive_vali_basedir='./data/vali/positive'
negative_vali_basedir='./data/vali/negative'

epoch=200
vali_split=0.3

index = 0
vali_index = 0

train_x_file_list = []
train_x = []
train_y = []

vali_x_file_list = []
vali_x=[]
vali_y = []
prob_y=[]

def create_x_y_mapping(train_or_vali):
    if(train_or_vali == 'train'):
        with open(train_mapping_file,'w') as f:
            f.write("file_path,label\n")
            for root,directs,filenames in os.walk(positive_train_basedir):
                for filename in filenames:
                    pathName=os.path.join(root,filename)
                    if( ('jpg' in pathName) or ('png' in pathName) ):
                        f.write(pathName+',1\n')
            for root,directs,filenames in os.walk(negative_train_basedir):
                for filename in filenames:
                    pathName=os.path.join(root,filename)
                    if( ('jpg' in pathName) or ('png' in pathName) ):
                        f.write(pathName+',0\n')
            for root,directs,filenames in os.walk(polluted_train_basedir):
                for filename in filenames:
                    pathName=os.path.join(root,filename)
                    if( ('jpg' in pathName) or ('png' in pathName) ):
                        f.write(pathName+',2\n')
    else:
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

def read_x_y_mapping(train_or_vali,shuffle):
    if(train_or_vali=='train'):
        global train_x_file_list
        global train_y
        train_x_file_list=[]
        train_y=[]
        if( not os.path.exists(train_mapping_file)):
            create_x_y_mapping('train')
        with open(train_mapping_file,'r') as f:
            next(f)
            lines=f.readlines()
            for line in lines:
                train_x_file_list.append(line.split(',')[0])
                train_y.append(line.split(',')[1][:-1])
        if(shuffle):
            c = list(zip(train_x_file_list, train_y))
            random.shuffle(c)
            train_x_file_list, train_y = zip(*c)
        train_y = np.array(train_y)
        train_y = np_utils.to_categorical(train_y,3)
    else:
        global vali_x_file_list
        global vali_y
        vali_x_file_list=[]
        vali_y=[]
        if( not os.path.exists(vali_mapping_file)):
            create_x_y_mapping('vali')
        with open(vali_mapping_file,'r') as f:
            next(f)
            lines=f.readlines()
            for line in lines:
                vali_x_file_list.append(line.split(',')[0])
                vali_y.append(line.split(',')[1][:-1])
        if(shuffle):
            c = list(zip(vali_x_file_list, vali_y))
            random.shuffle(c)
            vali_x_file_list, vali_y = zip(*c)
        vali_y = np.array(vali_y)
        vali_y = np_utils.to_categorical(vali_y,3)


def load_all_valid():
    global vali_x
    vali_x = np.zeros([len(vali_x_file_list), height, width, 3])
    for i,f in enumerate(vali_x_file_list):
        vali_x[i]=Image.open(f).resize([width,height])
    vali_x=vali_x.astype('float64')
    vali_x/=255.

def generate_valid_from_train():
    global train_x_file_list
    global train_y
    global vali_x_file_list
    global vali_y
    vali_x_file_list = train_x_file_list[ :math.ceil(len(train_x_file_list)*vali_split) ]
    vali_y = train_y [ :math.ceil(len(train_x_file_list)*vali_split) ]
    train_x_file_list = train_x_file_list [math.floor(len(train_x_file_list)*vali_split):]
    train_y = train_y [math.floor(len(train_x_file_list)*vali_split):]

def resize_preprocessing(data,label):
    data=data.resize([width,height])
    data = np.asarray(data)
    data = data.astype('float64')
    
    if (random.random() > 0.5 and int(label[1])==1):
        data = cv2.flip(data, 1)
    data/=255.
    return data

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

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.summary()
    return model

def training(model):
    es=EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    #rlr=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    mck=ModelCheckpoint(filepath='cnn_model_best.h5',monitor='val_loss',save_best_only=True)
    if(host=='ican-1080ti' and gpu == 'both'):
        model = multi_gpu_model(model, gpus=2)
    class_weight = {0: negative_weight,1: positive_weigt,2: polluted_weight}
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit_generator(data_generator(True),validation_data=(vali_x,vali_y),validation_steps=1,steps_per_epoch=len(train_x_file_list)//batch_size, epochs=epoch,callbacks=[mck,es],class_weight=class_weight)
    model.save('cnn_model.h5')
    
def predict():
    global vali_x
    global model_to_load
    global prob_y
    if(model_to_load==''):
        model_to_load='cnn_model_best.h5'
    model=load_model(model_to_load)
    model.summary()
    read_x_y_mapping('vali',False)
    load_all_valid()
    loss, accuracy = model.evaluate(vali_x, vali_y)
    print("loss: "+str(loss))
    print("accu: "+str(accuracy))
    prob_y = model.predict(vali_x)
    y_true=[]
    y_pred=[]
    with open('result.csv','w') as file:
        file.write("filename,real_value,pred_value\n")
        for i,p in enumerate(vali_x_file_list):
            file.write(p+","+str(np.argmax(vali_y[i]))+","+str(np.argmax(prob_y[i]))+'\n')
            print(str(i)+" "+ str(np.argmax(vali_y[i])) +" -> "+str(np.argmax(prob_y[i])))
    y_true=np.argmax(vali_y,axis=1)
    y_pred=np.argmax(prob_y,axis=1)
    evaluate(y_true,y_pred)
    labels=["negative", "positive", "polluted"]
    #plt.figure()
    #from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_true,y_pred)
    #plot_confusion_matrix(cm,classes=labels,title='Confusion matrix')
    #plt.show()

def saliency_map(mode,backprop_modifier=None,grad_modifier="absolute"):
    shutil.rmtree("./saliency_map/")
    os.mkdir("saliency_map")
    import keras.backend as K
    from vis.visualization import visualize_saliency, visualize_cam
    import scipy.ndimage as ndimage
    global vali_x
    global vali_y
    global prob_y
    global model_to_load
    file_list=[]
    dataset=os.sys.argv[3]
    portion=[]
    model=load_model(model_to_load)
    np.seterr(divide='ignore',invalid='ignore')
    
    if(dataset=='vali'):
        predict()
        amount=len(vali_x_file_list)
        file_list=vali_x_file_list
        
        # for i in range(len(vali_x_file_list)):
        #     if(vali_y[i][portion]==1.):
        #         print("Appending "+vali_x_file_list[i])
        #         file_list.append(vali_x_file_list[i])
        #         vali_x[n]=Image.open(vali_x_file_list[i]).resize([width,height])
        #         n+=1
        #     if(n>=amount):
        #         break
        # vali_x=vali_x.astype('float64')
        # vali_x/=255.
        
    if(dataset=='train'):
        portion=(int(os.sys.argv[4]))
        amount=int(os.sys.argv[5])
        read_x_y_mapping('train',True)
        vali_x = np.zeros([amount, height,width, 3])
        n=0
        for i in range(len(train_x_file_list)):
            if(train_y[i][portion]==1.):
                print("Appending "+train_x_file_list[i])
                file_list.append(train_x_file_list[i])
                vali_x[n]=(Image.open(train_x_file_list[i]).resize([width,height]))
                n+=1
            if(n>=amount):
                break
        vali_x=vali_x.astype('float64')
        vali_x/=255.

    input_img = model.input
    for i,img in enumerate(vali_x):
        if(dataset=='vali'):
            portion=np.argmax(prob_y[i])
        print("Creating saliency map of "+str(i)+' -> '+file_list[i]+', class='+str(portion))
        filename='./saliency_map/'+str(i)+'_predclass='+str(portion)+'_trueclass='+str(np.argmax(vali_y[i]))
        heatmap = visualize_cam(model, layer_idx=-1, filter_indices=portion, seed_input=img,backprop_modifier=backprop_modifier,grad_modifier=grad_modifier)
        plt.imshow(img)
        if(mode=='show'):
            plt.show()
        else:
            plt.savefig(filename+'.jpg',dpi=100)
        im1=plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        im2 = plt.imshow(heatmap,  alpha=.4, interpolation='bilinear')
        if(mode=='show'):
            plt.show()
        else:
            plt.savefig(filename+'_heatmap.jpg',dpi=100)
def main():
    if(mode=='train'):
        read_x_y_mapping('train',True)
        read_x_y_mapping('vali',False)
        load_all_valid()
        print(np.shape(vali_x))
        if(host=='ican-1080ti' and gpu =='both'):
            with tf.device('/cpu:0'):
                model=get_model()
        else:
            model=get_model()
        training(model)
        predict()
    elif(mode=='predict'):
        predict()
    elif(mode=='saliencymap'):
        saliency_map(mode='save')
if __name__ == "__main__":
    main()

