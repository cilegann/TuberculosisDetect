import os
import time
import tensorflow as tf
import numpy
import random
import numpy as np
from keras import *
from keras import utils as np_utils
from keras.layers import *
from keras.layers import multiply
from keras.layers.core import Lambda
from keras.models import Model,load_model,model_from_json
from keras import backend as K
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adam
from Capsule_Keras import *
from evaluate_tools import plot_confusion_matrix,evaluate
import keras.backend.tensorflow_backend as KTF
from utils import *


train_mapping_file='./data/YOLO9000_x_y_mapping.csv'
vali_mapping_file='./data/YOLO9000_vali_x_y_mapping.csv'
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

    model_input=Input(shape=(args.vector_length,))
    
    model_a=Dropout(0.25)(model_input)
    model_a=Dense(32,activation='relu')(model_a)
    model_a=BatchNormalization()(model_a)
    output_a=Dense(2,activation='softmax')(model_a)

    model_b=Dropout(0.25)(model_input)
    model_b=Dense(32,activation='relu')(model_b)
    model_b=BatchNormalization()(model_b)
    output_b=Dense(2,activation='softmax')(model_b)

    model_output=Concatenate()([output_a,output_b])
    def two_stage_classifier(x):
        import tensorflow as tf
        from keras.layers import multiply
        a1,a2,b1,b2=Lambda(lambda tensor: tf.split(tensor,4,1))(x)
        return K.concatenate([multiply([a1,b2]),a2,multiply([a1,b1])])
    model_output=Lambda(two_stage_classifier)(model_output)
    model=Model(model_input,model_output)

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
    print(" "*13,nowtime.replace("-","").replace(":",""))
    print("#########################################")
    scriptBackuper(os.path.basename(__file__),nowtime)
    jst=model.to_json()
    with open('./models/tstransferKeras_'+nowtime+'.json','w') as file:
        file.write(jst)
    cblog = CSVLogger('./log/tstransferKeras_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir=('./Graph/'+"tstransfernKeras_"+nowtime.replace("-","").replace(":","")),batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/tstransferKeras_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbckptw=ModelCheckpoint('./models/tstransferKeras_'+nowtime+'_best_weight.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbrlr=ReduceLROnPlateau()
    x_train_list,y_train,indexes=read_x_y_mapping(mappings,basedirs,'train',not args.balance,args,txt=True)
    x_vali_list,y_vali,_=read_x_y_mapping(mappings,basedirs,'vali',False,args,txt=True)
    x_vali=load_all_valid(x_vali_list,args,txt=True)
    
    try:
        model.fit_generator(
            data_generator(True,x_train_list,y_train,args,indexes,txt=True),
            validation_data=(x_vali,y_vali),
            validation_steps=1,
            #steps_per_epoch=(46),
            steps_per_epoch=min(np.asarray([indexes[i][2] for i in range(3)]))//(args.batch//3),
            #steps_per_epoch=int(len(x_train_list))//int(batch_size),
            epochs=args.epochs,
            callbacks=[cblog,cbtb,cbckpt,cbckptw],
            class_weight=([0.092,0.96,0.94] if not args.balance else [1,1,1])
        )
        model.save('./models/tstransferKeras_'+nowtime+'.h5')
        model.save_weights('./models/tstransferKeras_'+nowtime+'_weight.h5')
        
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
    x_vali_list,y_vali,_=read_x_y_mapping(mappings,basedirs,'vali',False,args,txt=True)
    x_vali=load_all_valid(x_vali_list,args,txt=True)
    y_pred=model.predict(x_vali)
    y_pred=np.argmax(y_pred,axis=1)
    y_ture=np.argmax(y_vali,axis=1)
    labels=['negative','positive','polluted']
    plot_confusion_matrix(y_ture,y_pred,labels)
    evaluate(y_ture,y_pred)

def dev(args):
    model=get_model(args)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Two stage transfer learning on TB")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Testing mode')
    parser.add_argument('--dev',action='store_true',help='Dev mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('--vector_length',type=int,default=173056)
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
        test(args)
    if args.dev:
        print("Dev mode")
