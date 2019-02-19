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
from keras.models import Model
from keras import backend as K
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from Capsule_Keras import *
from evaluate_tools import plot_confusion_matrix,evaluate

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

def read_x_y_mapping(train_or_vali,shuffle):
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

def data_generator(is_training,file_lists,y):
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

def load_all_valid(file_list):
    x_vali=np.zeros([len(file_list),height,width,num_of_classes])
    for i,f in enumerate(file_list):
        x_vali[i]=Image.open(f).resize([width,height])
    x_vali=x_vali.astype('float64')
    x_vali/=255.
    return x_vali

###################################################################################

def get_model(args):
    input_image = Input(shape=(None,None,3))
    cnn = Conv2D(32, (3, 3), activation='relu',data_format='channels_last')(input_image)
    cnn = Conv2D(32, (3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(num_of_classes, 16, args.routing, args.share)(cnn)
    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(10,))(capsule) #L2 norm of each capsule
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
    cbtb = TensorBoard(log_dir='./Graph',batch_size=batch_size)
    cbckpt=ModelCheckpoint('./models/capsule_'+nowtime+'_best.h5',monitor='val_loss',save_best_only=True)
    cbes=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2+0.5*(1-y_true)*K.relu(y_pred-0.1)**2,optimizer=Adam(),metrics=['accuracy','loss'])
    x_train_list,y_train=read_x_y_mapping('train',True)
    x_vali_list,y_vali=read_x_y_mapping('vali',False)
    x_vali=load_all_valid(x_vali_list)
    model.fit_generator(data_generator(True,x_train_list,y_train),
                        validation_data=(x_vali,y_vali),
                        validation_steps=1,
                        steps_per_epoch=len(x_train_list//batch_size),
                        epochs=args.epochs,
                        callbacks=[cblog,cbtb,cbckpt,cbes])
    model.save('./models/capsule_'+nowtime+'.h5')
    y_pred=model.predict(x_vali)
    y_pred=np.argmax(y_pred,axis=1)
    y_true=np.argmax(y_vali,axis=1)
    labels=['negative','positive','polluted']
    plot_confusion_matrix(y_true,y_pred,labels)
    evaluate(y_true,y_pred)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capsule Network on TB.")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_ture',help='Tesing mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('-r','--routing',type=int,help='#iteration of routing algorithm')
    parser.add_argument('-b','--batch',type=int,default=32,help='Batch size')
    parser.add_argument('-e','--epochs',type=int,default=200,help='#Epochs')
    parser.add_argument('-s','--share',action='store_true',help='Share weight or not')
    args=parser.parse_args()
    print(args)
    config_environment(args)
    if args.train:
        print("Train")
        train(args)
        