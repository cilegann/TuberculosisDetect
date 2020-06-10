# coding: utf-8

import tensorflow as tf
import os
import random
import math
import time

import keras
import keras.backend as K
from keras import utils as np_utils
from keras.preprocessing import image
from keras import Sequential
from keras.models import load_model,Model
from keras.utils import multi_gpu_model
from keras.layers import *
from keras.optimizers import *
import keras.losses
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,CSVLogger,ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF

import numpy as np
from utils import *
from evaluate_tools import cam,plot_confusion_matrix,evaluate

def config_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)

def get_model(args):
    model = load_model('./src/yolo9000_yad2k.h5')
    for i,l in enumerate(model.layers):
        l.trainable=False
        model.layers[i].name='yolo_'+l.name
    hidden=Flatten()(model.get_layer('yolo_leaky_re_lu_18').output)
    hidden=Dropout(0.5)(hidden)
    hidden=Dense(32,activation='relu')(hidden)
    hidden=BatchNormalization()(hidden)
    x=Dense(args.n_labels, activation='softmax')(hidden)
    model=Model(model.input,x)

    model.summary()
    return model

###################################################################################

def train(model):
    model=get_model(args)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    if not os.path.exists('./log'):
        os.mkdir('./log')
    nowtime=time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    print("######### TRAINING FILE POSTFIX #########")
    print(" "*13,nowtime)
    print("#########################################")
    scriptBackuper(os.path.basename(__file__),nowtime)
    jst=model.to_json()
    with open('./models/transfer_yad2k_'+nowtime+'_json.json','w') as file:
        file.write(jst)
    cblog = CSVLogger('./log/transfer_yad2k_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir=('./Graph/'+"transfer_yad2k_"+nowtime.replace("-","").replace(":","")),batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/transfer_yad2k_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbckptw=ModelCheckpoint('./models/transfer_yad2k_'+nowtime+'_best_weight.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbrlr=ReduceLROnPlateau(monitor='val_loss',patience=5)
    x_train_list,y_train,indexes=read_mapping(args.mappings[0],not args.balance,args)
    x_vali_list,y_vali,_=read_mapping(args.mappings[1],False,args)
    x_vali=load_all_valid(x_vali_list,args)
    try:
        model.fit_generator(
            data_generator(True,x_train_list,y_train,args,indexes),
            validation_data=(x_vali,y_vali),
            validation_steps=1,
            #steps_per_epoch=(46),
            steps_per_epoch=min(np.asarray([indexes[i][2] for i in range(3)]))//(args.batch//3) if args.balance else int(len(x_train_list))//int(args.batch),
            #steps_per_epoch=int(len(x_train_list))//int(batch_size),
            epochs=args.epochs,
            callbacks=[cblog,cbtb,cbckpt,cbckptw,cbes,cbrlr]
            #class_weight=([0.092,0.96,0.94] if not args.balance else [1,1,1])
        )
        model.save('./models/transfer_yad2k_'+nowtime+'.h5')
        model.save_weights('./models/transfer_yad2k_'+nowtime+'_weight.h5')
        
        y_pred=model.predict(x_vali)
        y_pred=np.argmax(y_pred,axis=1)
        y_ture=np.argmax(y_vali,axis=1)
        labels=['negative','positive','polluted']
        plot_confusion_matrix(y_ture,y_pred,labels)
        evaluate(y_ture,y_pred)
    except KeyboardInterrupt:
        os.system("sh purge.sh "+nowtime)
def test(args):
    model=load_model(args.model)
    x_vali_list,y_vali,_=read_mapping(args.mappings[1],False,args)
    x_vali=load_all_valid(x_vali_list,args)
    y_pred=model.predict(x_vali)
    y_pred=np.argmax(y_pred,axis=1)
    y_ture=np.argmax(y_vali,axis=1)
    labels=['negative','positive','polluted']
    plot_confusion_matrix(y_ture,y_pred,labels)
    evaluate(y_ture,y_pred)

def dev(args):
    pass

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Transfer learning on TB")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Testing mode')
    parser.add_argument('--dev',action='store_true',help='Dev mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('--best',action='store_true',help='Load best model or not')
    parser.add_argument('--width',type=int,default=416)
    parser.add_argument('--height',type=int,default=416)
    parser.add_argument('--batch',type=int,default=64,help='Batch size')
    parser.add_argument('--epochs',type=int,default=200,help='#Epochs')
    parser.add_argument('--balance',action='store_true',help='Balance data by undersampling the majiroty data')
    parser.add_argument('--n_labels',type=int,default=3)
    parser.add_argument('--gpu',type=str,default='1',help='No. of GPU to use')
    parser.add_argument('--data',type=str,default='190410_newdata_smote',help="Dataset")
    parser.add_argument('--augment',action='store_true',help='Data augment by randomly flipping image')
    args=parser.parse_args()
    config_environment(args)
    
    train_mapping_file='./mapping/'+args.data+'_train_cnn_mapping.csv'
    vali_mapping_file='./mapping/'+args.data+'_vali_cnn_mapping.csv'
    args.mappings=[train_mapping_file,vali_mapping_file]

    # polluted_train_basedir=os.path.join(data,'polluted')
    # positive_train_basedir=os.path.join(data,'positive')
    # negative_train_basedir=os.path.join(data,'negative')
    # polluted_vali_basedir=os.path.join(data,'vali/polluted')
    # positive_vali_basedir=os.path.join(data,'vali/positive')
    # negative_vali_basedir=os.path.join(data,'vali/negative')
    # args.basedirs=[polluted_train_basedir,positive_train_basedir,negative_train_basedir,polluted_vali_basedir,positive_vali_basedir,negative_vali_basedir]

    if args.train:
        print("Training mode")
        if args.balance:
            args.batch-=(args.batch%3)
        train(args)

    if args.test:
        print("Testing mode")
        if args.model==None:
            print("Please specify model with -m or --model")
        else:
            if 'h5' not in args.model:
                for r,_,fs in os.walk('./models'):
                    for f in fs:
                        if args.model in f:
                            if 'best.h5' in f and args.best:
                                args.model=os.path.join(r,f)
                                print("Model:",args.model)
                            elif 'best' not in f and '.h5' in f and not args.best:
                                args.model=os.path.join(r,f)
                                print("Model:",args.model)
            test(args)
    if args.dev:
        print("Dev mode")

