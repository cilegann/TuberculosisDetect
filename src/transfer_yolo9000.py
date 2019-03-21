# coding: utf-8

import tensorflow as tf
import os
import random
import math
import time

import keras
import keras.backend as K
from keras import utils as np_utils
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


vector_length=173056
train_mapping_file='./data/YOLO9000_x_y_mapping.csv'
vali_mapping_file='./data/YOLO9000_vali_x_y_mapping.csv'
mapping_files=[train_mapping_file,vali_mapping_file]
polluted_train_basedir='./data/polluted'
positive_train_basedir='./data/positive'
negative_train_basedir='./data/negative'
polluted_vali_basedir='./data/vali/polluted'
positive_vali_basedir='./data/vali/positive'
negative_vali_basedir='./data/vali/negative'
basedirs=[polluted_train_basedir,positive_train_basedir,negative_train_basedir,polluted_vali_basedir,positive_vali_basedir,negative_vali_basedir]

def config_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)

def get_model(args):
    input_layer=Input(shape=(args.vector_length,))
    hidden=Dropout(0.25)(input_layer)
    hidden=Dense(32,activation='relu')(hidden)
    output=Dense(args.n_labels,activation='softmax')(hidden)
    model=Model(input_layer,output)
    return model

###################################################################################

def training(model):
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
    with open('./models/transferyolo_'+nowtime+'_json.json','w') as file:
        file.write(jst)
    cblog = CSVLogger('./log/transfer_yolo_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir=('./Graph/'+"transfer_yolo_"+nowtime.replace("-","").replace(":","")),batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/transfer_yolo_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbckptw=ModelCheckpoint('./models/transfer_yolo_'+nowtime+'_best_weight.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbrlr=ReduceLROnPlateau()
    x_train_list,y_train,indexes=read_x_y_mapping(mappings,basedirs,'train',not args.balance,args,txt=True)
    x_vali_list,y_vali,_=read_x_y_mapping(mappings,basedirs,'vali',False,args)
    x_vali=load_all_valid(x_vali_list,args)
    try:
        model.fit_generator(
            data_generator(True,x_train_list,y_train,args,indexes),
            validation_data=(x_vali,y_vali),
            validation_steps=1,
            steps_per_epoch=(46),
            #steps_per_epoch=min(np.asarray([indexes[i][2] for i in range(3)]))//args.batch,
            #steps_per_epoch=int(len(x_train_list))//int(batch_size),
            epochs=args.epochs,
            callbacks=[cblog,cbtb,cbckpt],
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

###################################################################################

def predict():
    global vali_x
    global model_to_load
    global prob_y
    global model_to_load
    if(model_to_load==''):
        model_to_load='yolo9000_model_best.h5'
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
    # for y in prob_y:
    #     if y[1]>0.33:
    #         y_pred.append(1)
    #     else:
    #         y_pred.append(np.argmax(y))
    labels=["Negative", "Positive", "Polluted"]
    plot_confusion_matrix(y_true,y_pred,classes=labels)
    evaluate(y_true,y_pred)
    return y_true,y_pred

        
###################################################################################

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def main():
    #TODO: arg.vector_length
    if(mode=='train'):
        if 'balance' in os.sys.argv:
            read_x_y_mapping('train',False)
        else:
            read_x_y_mapping('train',True)
        read_x_y_mapping('vali',False)
        load_all_valid()
        print(np.shape(vali_x))
        model=get_model()
        print(get_model_memory_usage(batch_size,model))
        training(model)
        predict()
    elif(mode=='predict'):
        y_true,y_pred=predict()

    elif(mode=='cam'):
        model=load_model(model_to_load)
        imgs=[]
        y_true,y_pred=predict()
        for i,img in enumerate(imgs):
            cam(str(i)+'_'+y_pred[i],img,y_pred[i],model)
    else:
        read_x_y_mapping('train',False)
        
        print(train_start_index)
        print(train_end_index)
        print(train_len)
        data_generator_balance(True)
if __name__ == "__main__":
    main()

