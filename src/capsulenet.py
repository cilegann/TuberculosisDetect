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
from evaluate_tools import evaluate,plot_confusion_matrix,plot_confusion_matrix_by_cm
import keras.backend.tensorflow_backend as KTF
from utils import *

train_mapping_file='./data/CNN_x_y_mapping.csv'
vali_mapping_file='./data/CNN_vali_x_y_mapping.csv'
mappings=[train_mapping_file,vali_mapping_file]

polluted_train_basedir='./data/polluted'
positive_train_basedir='./data/positive'
negative_train_basedir='./data/negative'
polluted_vali_basedir='./data/vali/polluted'
positive_vali_basedir='./data/vali/positive'
negative_vali_basedir='./data/vali/negative'
basedirs=[polluted_train_basedir,positive_train_basedir,negative_train_basedir,polluted_vali_basedir,positive_vali_basedir,negative_vali_basedir]


def config_environment(args):
    global batch_size
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def marginLoss(y_true,y_pred):
    return y_true*K.relu(0.9-y_pred)**2+0.5*(1-y_true)*K.relu(y_pred-0.1)**2

def get_model(args):
    input_image = Input(shape=(None,None,3))
    cnn = Conv2D(64, (3, 3), activation='relu',data_format='channels_last')(input_image)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = AveragePooling2D((2,2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(8, 16, args.routing, args.share)(cnn)
    capsule = Capsule(args.n_labels,16,args.routing,args.share)(capsule)
    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(args.n_labels,))(capsule) #L2 norm of each capsule
    model = Model(inputs=input_image, outputs=output)

    model.summary()

    return model

###################################################################################

def train(args):
    model=get_model(args)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    nowtime=time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    cblog = CSVLogger('./log/capsule_'+nowtime+'.csv')
    cbtb = TensorBoard(log_dir='./Graph',batch_size=args.batch)
    cbckpt=ModelCheckpoint('./models/capsule_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.compile(loss=marginLoss,optimizer=Adam(),metrics=['accuracy'])
    x_train_list,y_train,indexes = read_x_y_mapping(mappings,basedirs,'train',True,args)
    x_vali_list,y_vali,_ = read_x_y_mapping(mappings,basedirs,'vali',True,args)
    x_vali=load_all_valid(x_vali_list,args)
    model.fit_generator(data_generator(True,x_train_list,y_train,args,indexes),
                        validation_data=(x_vali,y_vali),
                        validation_steps=1,
                        steps_per_epoch=(len(x_train_list)//args.batch),
                        epochs=args.epochs,
                        callbacks=[cblog,cbtb,cbckpt,cbes])
    model.save('./models/capsule_'+nowtime+'.h5')
    model.save_weights('./models/capsule_'+nowtime+'_weight.h5')
    y_pred=model.predict(x_vali)
    y_pred=np.argmax(y_pred,axis=1)
    y_true=np.argmax(y_vali,axis=1)
    labels=['negative','positive','polluted']
    plot_confusion_matrix(y_true,y_pred,labels)
    evaluate(y_true,y_pred)
    
###################################################################################

def test(args):
    model=load_model(args.model)
    x_vali_list,y_vali,_=read_x_y_mapping(mappings,basedirs,'vali',False,args)
    x_vali=load_all_valid(x_vali_list,args)
    y_pred=model.predict(x_vali)
    y_pred=np.argmax(y_pred,axis=1)
    y_true=np.argmax(y_vali,axis=1)
    plot_confusion_matrix(y_true,y_pred,["neg","pos","pol"])
    evaluate(y_true,y_pred)
###################################################################################

def dev(args):
    model=get_model(args)
    
    model.compile(loss=marginLoss,optimizer=Adam(),metrics=['accuracy'])
    model.save("all.h5")

    model=None
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Capsule': Capsule,'marginLoss':marginLoss})
    loaded_model.load_weights("weight.h5")
    model.summary()
    model=load_model("all.h5", custom_objects={'Capsule': Capsule,'marginLoss':marginLoss})
    model.summary()
    pass

###################################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capsule Network on TB.")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Tesing mode')
    parser.add_argument('--dev',action='store_true',help='Dev testing mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('-r','--routing',type=int,default=3,help='#iteration of routing algorithm')
    parser.add_argument('-b','--batch',type=int,default=32,help='Batch size')
    parser.add_argument('-e','--epochs',type=int,default=200,help='#Epochs')
    parser.add_argument('-s','--share',action='store_true',help='Share weight or not')
    parser.add_argument('--width',type=int,default=420)
    parser.add_argument('--height',type=int,default=131)
    parser.add_argument('--balance',action='store_true',help="Balance data by undersampling the majiroty data")
    parser.add_argument('--n_labels',type=int,default=3)
    args=parser.parse_args()
    print(args)
    config_environment(args)
    if args.train:
        print("Training mode")
        if args.balance:
            args.batch-=(args.batch%3)
            print("Under balance mode, change batch size to multiple of 3:",args.batch)
        train(args)
    if args.test:
        if args.model is not None:
            print("Testing model: ",args.model)
            test(args)
        else:
            print("Please specify the model you want to test on with '-m model_path'")
    if args.dev:
        print("Dev testing mode")
        dev(args)