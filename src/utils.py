import os
import random
from keras import utils as np_utils
import numpy as np
import cv2
from PIL import Image
def create_x_y_mapping(mappings,basedirs,train_or_vali):
    train_mapping_file=mappings[0]
    vali_mapping_file=mappings[1]

    polluted_train_basedir=basedirs[0]
    positive_train_basedir=basedirs[1]
    negative_train_basedir=basedirs[2]
    polluted_vali_basedir=basedirs[3]
    positive_vali_basedir=basedirs[4]
    negative_vali_basedir=basedirs[5]
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

def read_x_y_mapping(mappings,basedirs,train_or_vali,shuffle,args):
    train_mapping_file=mappings[0]
    vali_mapping_file=mappings[1]
    file_list=[]
    y=[]
    if(train_or_vali=='train'):
        mapping_file=train_mapping_file
    else:
        mapping_file=vali_mapping_file
    if(not os.path.exists(mapping_file)):
        create_x_y_mapping(mappings,basedirs,train_or_vali)
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
        indexes=[-1,-1,-1]
    else:
        s0=y.index('0')
        s1=y.index('1')
        s2=y.index('2')
        e0=s1-1
        e1=s2-1
        e2=len(y)-1
        l0=e0-s0+1
        l1=e1-s1+1
        l2=e2-s2+1
        indexes=[[s0,e0,l0],[s1,e1,l1],[s2,e2,l2]]
    return file_list,np_utils.to_categorical(np.array(y),args.n_labels),indexes

def preprocessing_augment(data,label,args):
    data=data.resize([args.width,args.height])
    data = np.asarray(data)
    data = data.astype('float64')
    if (random.random() > 0.5 and int(label[1])==1):
        data = cv2.flip(data, 1)
    data/=255.
    return data

###################################################################################

train_index=0
vali_index=0
train_indexes=[-1,-1,-1]

def data_generator(is_training,file_lists,y,args,indexes=None):
    is_balanced=args.balance
    batch_size=args.batch
    height=args.height
    width=args.width
    if is_balanced:
        global train_indexes
        if train_indexes==[-1,-1,-1]:
            train_indexes=[indexes[i][0] for i in range(3)]
        while(1):
            if is_training:
                file_list=[]
                label_list=[]
                for i in range(3):
                    # flag: train_indexes[i]
                    # start: indexes[i][0]
                    # end: indexes[i][1]
                    # length: indexes[i][2]
                    for n in range(batch_size/3):
                        file_list.append(file_lists[ train_indexes[i] ])
                        label_list.append(y[ train_indexes[i] ])
                        train_indexes[i]+=1
                        if(train_indexes[i]>indexes[i][1]):
                            train_indexes[i]=indexes[i][0]
                label_list=np.asarray(label_list)
                output = np.zeros([batch_size, height,width, 3])
                for i in range(batch_size):
                    output[i]=preprocessing_augment(Image.open(file_list[i]),label_list[i],args)
                yield output, label_list
            
    else:
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
                output[i]=preprocessing_augment(Image.open(file_list[i]),label_list[i],args)
            yield output, label_list

###################################################################################

def load_all_valid(file_list,args):
    x_vali=np.zeros([len(file_list),args.height,args.width,args.n_labes])
    for i,f in enumerate(file_list):
        x_vali[i]=Image.open(f).resize([args.width,args.height])
    x_vali=x_vali.astype('float64')
    x_vali/=255.
    return x_vali

###################################################################################
