import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.backend as K
from keras import utils as np_utils
from keras.optimizers import rmsprop,adam
import keras.losses
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import keras.backend.tensorflow_backend as KTF
import random
import numpy as np
epoch=200
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

batch_size=32
index=0

def generator(train_or_not,lines):
    global index
    while(1):
        labels=np.zeros(batch_size)
        vectors=[]
        for i in range(batch_size):
            if(index>=300):
                index=0
            line=lines[index][:-1].split(",")
            if('negative' in line[0]):
                labels[i]=0
            elif('positive' in line[0]):
                labels[i]=1
            elif('polluted' in line[0]):
                labels[i]=2
            vectors.append(list(map(float, line[1].split(" "))))
            index+=1
        labels=np_utils.to_categorical(labels,3)
        vectors=np.asarray(vectors)
        yield vectors,labels

def get_model():
    model=Sequential()
    model.add(Dense(1024, input_shape=(173056,)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.summary()
    return model

def train(model,lines):
    es=EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    #rlr=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    mck=ModelCheckpoint(filepath='dnn_model_best.h5',monitor='loss',save_best_only=True)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit_generator(generator(True,lines),steps_per_epoch=len(lines)//batch_size, epochs=epoch,callbacks=[mck,es])
    model.save('dnn_model.h5')

def main():
    with open('./feature_vector.txt','r') as file:
        lines=file.readlines()
    random.shuffle(lines)
    train(get_model(),lines)

if __name__ == "__main__":
    main()

