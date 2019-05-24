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
from keras.models import Model,load_model,model_from_json
from keras import backend as K
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adam
from Capsule_Keras import *
from evaluate_tools import plot_confusion_matrix,evaluate
import keras.backend.tensorflow_backend as KTF
from utils import *
from keras.preprocessing.image import ImageDataGenerator


def config_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    

def get_model(args):
    model = Sequential()

    model.add(Conv2D(32,(3,3),strides=(1,1),input_shape=(args.height,args.width,3),data_format='channels_last'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    # model.add(Conv2D(32,(3,3),strides=(1,1)))
    # model.add(Activation('relu'))
    # #model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))


    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    # model.add(Conv2D(64,(3,3),strides=(1,1)))
    # model.add(Activation('relu'))
    # #model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))


    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.summary()
    return model

def train(args):
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
    with open('./models/cnn_'+nowtime+'_json.json','w') as file:
        file.write(jst)
    cblog = CSVLogger('./log/cnn_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir=('./Graph/'+"cnn_"+nowtime.replace("-","").replace(":","")),batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/cnn_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbckptw=ModelCheckpoint('./models/cnn_'+nowtime+'_best_weight.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbrlr=ReduceLROnPlateau()
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
            callbacks=[cblog,cbtb,cbckpt,cbckptw],
            class_weight=([0.092,0.96,0.94] if not args.balance else [1,1,1])
        )
        model.save('./models/cnn_'+nowtime+'.h5')
        model.save_weights('./models/cnn_'+nowtime+'_weight.h5')
        
        y_pred=model.predict(x_vali)
        y_pred=np.argmax(y_pred,axis=1)
        y_ture=np.argmax(y_vali,axis=1)
        labels=['negative','positive','polluted']
        plot_confusion_matrix(y_ture,y_pred,labels)
        evaluate(y_ture,y_pred)
    except KeyboardInterrupt:
        os.system("sh purge.sh "+nowtime)
    
def train_on_positive(args):
    model=get_model(args)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    if not os.path.exists('./log'):
        os.mkdir('./log')
    nowtime=time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    print("######### TRAINING FILE POSTFIX #########")
    print(" "*13,nowtime)
    print("#########################################")
    scriptBackuper(os.path.basename(__file__),nowtime)
    cblog = CSVLogger('./log/cnn_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir=('./Graph/'+"cnn_"+nowtime),batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/cnn_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbckptw=ModelCheckpoint('./models/cnn_'+nowtime+'_best_weight.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbrlr=ReduceLROnPlateau()
    x_train_list,y_train,t_indexes=read_mapping(args.mappings[0],not args.balance,args)
    x_vali_list,y_vali,v_indexes=read_mapping(args.mappings[1],False,args)
    x_train_list=x_train_list[t_indexes[1][0]:t_indexes[1][1]+1]
    y_train=y_train[t_indexes[1][0]:t_indexes[1][1]+1]
    x_vali_list=x_vali_list[v_indexes[1][0]:v_indexes[1][1]+1]
    y_vali=y_vali[v_indexes[1][0]:v_indexes[1][1]+1]
    x_train=load_all_valid(x_train_list,args)
    x_vali=load_all_valid(x_vali_list,args)
    try:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=False,  # randomly flip images
            vertical_flip=True,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        model.fit_generator(
            datagen.flow(x_train, y_train,
                        batch_size=args.batch),
            validation_data=(x_vali,y_vali),
            validation_steps=1,
            steps_per_epoch=15,
            epochs=10,
            callbacks=[cblog,cbtb,cbckpt],
            class_weight=([1,1,1])
        )
    except KeyboardInterrupt:
        os.system("sh purge.sh "+nowtime)
    return model,nowtime

def train_on_all(args,model,nowtime):
    x_train_list,y_train,indexes=read_mapping(args.mappings[0],False,args)
    x_vali_list,y_vali,_=read_mapping(args.mappings[1],False,args)
    x_vali=load_all_valid(x_vali_list,args)
    try:
        model.fit_generator(
            data_generator(True,x_train_list,y_train,args,indexes),
            validation_data=(x_vali,y_vali),
            validation_steps=1,
            steps_per_epoch=(15),
            epochs=args.epochs,
            callbacks=[cblog,cbtb,cbckpt],
            class_weight=([0.092,0.96,0.94] if not args.balance else [1,1,1])
        )
        model.save('./models/cnn_'+nowtime+'.h5')
        model.save_weights('./models/cnn_'+nowtime+'_weight.h5')
        jst=model.to_json()
        with open('./models/cnn_'+nowtime+'_json.h5','w') as file:
            file.write(jst)
        
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

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="CNN on TB")
    parser.add_argument('--tstrain',action='store_true',help='Training mode (positove first)')
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Testing mode')
    parser.add_argument('--dev',action='store_true',help='Dev mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('--best',action='store_true',help='Load best model or not')
    parser.add_argument('--width',type=int,default=420)
    parser.add_argument('--height',type=int,default=131)
    parser.add_argument('--batch',type=int,default=32,help='Batch size')
    parser.add_argument('--epochs',type=int,default=200,help='#Epochs')
    parser.add_argument('--balance',action='store_true',help='Balance data by undersampling the majiroty data')
    parser.add_argument('--n_labels',type=int,default=3)
    parser.add_argument('--gpu',type=str,default='1',help='No. of GPU to use')
    parser.add_argument('--data',type=str,default='190408_newdata',help='Dataset')
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
    # rgs.basedirs=[polluted_train_basedir,positive_train_basedir,negative_train_basedir,polluted_vali_basedir,positive_vali_basedir,negative_vali_basedir]

    if args.train:
        print("TS Training mode")
        if args.balance:
            args.batch-=(args.batch%3)
        train(args)
    if args.tstrain:
        print("Training mode")
        if args.balance:
            args.batch-=(args.batch%3)
        model,nowtime=train_on_positive(args)
        train_on_all(args,model,nowtime)
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
