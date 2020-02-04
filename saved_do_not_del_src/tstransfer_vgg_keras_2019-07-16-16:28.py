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
from keras.applications.vgg16 import VGG16

def config_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    batch_size=args.batch
    

def get_model(args):
    model_input=Input(shape=(224,224,3))

    model_a = VGG16(weights='imagenet', include_top=True)
    for l in model_a.layers:
        l.trainable=False
    model_a.layers.pop()
    model_a.layers.pop()
    hidden=Dropout(0.25)(model_a.layers[-2].output)
    hidden=Dense(32,activation='relu')(hidden)
    # hidden=BatchNormalization()(hidden)
    output_a=Dense(2, activation='softmax')(hidden)
    model_a=Model(model_a.input,output_a)

    model_b = VGG16(weights='imagenet', include_top=True)
    for l in model_b.layers:
        l.trainable=False
    model_b.layers.pop()
    model_b.layers.pop()
    hidden=Dropout(0.25)(model_b.layers[-2].output)
    hidden=Dense(32,activation='relu')(hidden)
    # hidden=BatchNormalization()(hidden)
    output_b=Dense(2, activation='softmax')(hidden)
    model_b=Model(model_b.input,output_b)

    output_a=model_a(model_input)
    output_b=model_b(model_input)
    model_output=Concatenate()([output_a,output_b])
    def two_stage_classifier(x):
        import tensorflow as tf
        from keras.layers import multiply
        a1,a2,b1,b2=Lambda(lambda tensor: tf.split(tensor,4,1))(x)
        return K.concatenate([a2,multiply([a1,b1]),multiply([a1,b2])])
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
    with open('./models/tstransfer_vgg_keras_'+nowtime+'.json','w') as file:
        file.write(jst)
    cblog = CSVLogger('./log/tstransfer_vgg_keras_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir=('./Graph/'+"tstransfer_vgg_keras_"+nowtime.replace("-","").replace(":","")),batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/tstransfer_vgg_keras_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbckptw=ModelCheckpoint('./models/tstransfer_vgg_keras_'+nowtime+'_best_weight.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbrlr=ReduceLROnPlateau(monitor='val_loss',verbose=1,patience=5)
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
            callbacks=[cblog,cbtb,cbckpt,cbckptw,cbrlr]
            #class_weight=([0.092,0.96,0.94] if not args.balance else [1,1,1])
        )
        model.save('./models/tstransfer_vgg_keras_'+nowtime+'.h5')
        model.save_weights('./models/tstransfer_vgg_keras_'+nowtime+'_weight.h5')
        
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
    model=get_model(args)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Two stage transfer learning on TB")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Testing mode')
    parser.add_argument('--dev',action='store_true',help='Dev mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('--best',action='store_true',help='Load best model or not')
    parser.add_argument('--width',type=int,default=224)
    parser.add_argument('--height',type=int,default=224)
    parser.add_argument('--batch',type=int,default=32,help='Batch size')
    parser.add_argument('--epochs',type=int,default=200,help='#Epochs')
    parser.add_argument('--balance',action='store_true',help='Balance data by undersampling the majiroty data')
    parser.add_argument('--n_labels',type=int,default=3)
    parser.add_argument('--gpu',type=str,default='1',help='No. of GPU to use')
    parser.add_argument('--data',type=str,default='190410_newdata_smote',help='Dataset')
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
        print(args.model)
