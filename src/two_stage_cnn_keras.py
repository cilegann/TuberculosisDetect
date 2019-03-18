import os
import time
import tensorflow as tf
import numpy
import random
import numpy as np
import cv2
from PIL import Image
from keras import *
from keras import utils as np_utils
from keras.layers import *
from keras.layers.core import Lambda
from keras.models import Model,load_model,model_from_json
from keras import backend as K
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from Capsule_Keras import *
from evaluate_tools import plot_confusion_matrix,evaluate
import keras.backend.tensorflow_backend as KTF
from utils import *

width=420
height=131
num_of_classes=3
batch_size=32
train_mapping_file='./data/cnn_x_y_mapping.csv'
vali_mapping_file='./data/cnn_vali_x_y_mapping.csv'
mappings=[train_mapping_file,vali_mapping_file]

polluted_train_basedir='./data/polluted'
positive_train_basedir='./data/positive'
negative_train_basedir='./data/negative'
polluted_vali_basedir='./data/vali/polluted'
positive_vali_basedir='./data/vali/positive'
negative_vali_basedir='./data/vali/negative'
basedirs=[polluted_train_basedir,positive_train_basedir,negative_train_basedir,polluted_vali_basedir,positive_vali_basedir,negative_vali_basedir]

def config_environment(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    batch_size=args.batch
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_model(args):

    model_input=Input(shape=(args.height,args.width,3))
    
    cnn_a=Conv2D(32,(3,3),activation='relu',data_format='channels_last',padding='same')(model_input)
    cnn_a=Conv2D(32,(3,3),activation='relu')(cnn_a)
    cnn_a=MaxPool2D((2,2))(cnn_a)
    cnn_a=Dropout(0.25)(cnn_a)
    cnn_a=Conv2D(64,(3,3),activation='relu')(cnn_a)
    cnn_a=Conv2D(64,(3,3),activation='relu')(cnn_a)
    cnn_a=Dropout(0.25)(MaxPool2D(64,(3,3))(cnn_a))
    dense_a=Flatten()(cnn_a)
    dense_a=Dropout(0.5)(Dense(128,activation='relu')(dense_a))
    dense_a=Dense(64,activation='relu')(dense_a)
    output_a=Dense(2,activation='softmax')(dense_a)
    
    cnn_b=Conv2D(32,(3,3),activation='relu',data_format='channels_last',padding='same')(model_input)
    cnn_b=Conv2D(32,(3,3),activation='relu')(cnn_b)
    cnn_b=MaxPool2D((2,2))(cnn_b)
    cnn_b=Dropout(0.25)(cnn_b)
    cnn_b=Conv2D(64,(3,3),activation='relu')(cnn_b)
    cnn_b=Conv2D(64,(3,3),activation='relu')(cnn_b)
    cnn_b=Dropout(0.25)(MaxPool2D(64,(3,3))(cnn_b))
    dense_b=Flatten()(cnn_b)
    dense_b=Dropout(0.5)(Dense(128,activation='relu')(dense_b))
    dense_b=Dense(64,activation='relu')(dense_b)
    output_b=Dense(2,activation='softmax')(dense_b)

    model_output=Lambda(lambda x,y:[x[0],x[1]*y[0],x[1]*y[1]])(output_a,output_b)
    model=Model(model_input,model_output)

    model.summary()
    return model

def train(args):
    model=get_model(args)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    if not os.path.exists('./log'):
        os.mkdir('./log')
    nowtime=time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    cblog = CSVLogger('./log/tscnnKeras_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir='./Graph',batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/tscnnKeras_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    
    x_train_list,y_train,indexes=read_x_y_mapping(mappings,basedirs,'train',False,args)
    x_vali_list,y_vali,_=read_x_y_mapping(mappings,basedirs,'vali',False,args)
    x_vali=load_all_valid(x_vali_list,args)
    
    model.fit_generator(
        data_generator(True,x_train_list,y_train,args,indexes),
        validation_data=(x_vali,y_vali),
        validation_steps=1,
        steps_per_epoch=(46),
        epochs=args.epochs,
        callbacks=[cblog,cbtb,cbckpt,cbes]
    )
    model.save('./models/tscnnKeras_'+nowtime+'.h5')
    model.save_weights('./models/tscnnKeras_'+nowtime+'_weight.h5')
    jst=model.to_json()
    with open('./models/tscnnKeras_'+nowtime+'_json.h5','w') as file:
        file.write(jst)
    
    y_pred=model.predict(x_vali)
    y_pred=np.argmax(y_pred,axis=1)
    y_ture=np.argmax(y_vali,axis=1)
    labels=['negative','positive','polluted']
    plot_confusion_matrix(y_ture,y_pred,labels)
    evaluate(y_ture,y_pred)

def test():
    pass

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="cnn on TB")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Testing mode')
    parser.add_argument('--dev',action='store_true',help='Dev mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('--width',type=int,default=420)
    parser.add_argument('--height',type=int,default=131)
    parser.add_argument('--batch',type=int,default=32,help='Batch size')
    parser.add_argument('--epochs',type=int,default=200,help='#Epochs')
    parser.add_argument('--balance',action='store_true',help='Balance data by undersampling the majiroty data')
    parser.add_argument('--n_labels',type=int,default=3)
    args=parser.parse_args()
    config_environment(args)
    if args.train:
        print("Training mode")
        if args.balance:
            args.batch-=(args.batch%3)
        train(args)

    if args.test:
        print("Testing mode")
    if args.dev:
        print("Dev mode")
