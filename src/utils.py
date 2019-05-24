import os
import random
from keras import utils as np_utils
import numpy as np
import cv2
from PIL import Image
from random import randint, uniform,choice
import random
import string
import shutil
import math
from tqdm import tqdm
# this is v1 mapping creator
def create_x_y_mapping(mappings,basedirs,train_or_vali,txt=False):

    train_mapping_file=next(c for c in mappings if "vali" not in c)
    vali_mapping_file=next(c for c in mappings if "vali" in c)

    polluted_train_basedir=next(c for c in basedirs if "polluted" in c and "vali" not in c)
    positive_train_basedir=next(c for c in basedirs if "positive" in c and "vali" not in c)
    negative_train_basedir=next(c for c in basedirs if "negative" in c and "vali" not in c)
    polluted_vali_basedir=next(c for c in basedirs if "polluted" in c and "vali" in c)
    positive_vali_basedir=next(c for c in basedirs if "positive" in c and "vali" in c)
    negative_vali_basedir=next(c for c in basedirs if "negative" in c and "vali" in c)

    basedir_list=[]
    if(train_or_vali=='train'):
        mapping_file=train_mapping_file
        basedir_list=[negative_train_basedir,positive_train_basedir,polluted_train_basedir]
    else:
        mapping_file=vali_mapping_file
        basedir_list=[negative_vali_basedir,positive_vali_basedir,polluted_vali_basedir]
    print(" >Creating mapping file:",mapping_file)
    print("  NTB:",negative_train_basedir)
    print("  PTB:",positive_train_basedir)
    print("  XTB:",polluted_train_basedir)
    print("  NVB:",negative_vali_basedir)
    print("  PVB:",positive_vali_basedir)
    print("  XVB:",polluted_vali_basedir)
    with open(mapping_file,'w') as f:
        f.write("file_path,label\n")
        for i,b in enumerate(basedir_list):
            for root, directs,filenames in os.walk(b):
                for filename in filenames:
                    if not txt:
                        if 'txt' not in filename:
                            pathName=os.path.join(root,filename)
                            if( ('jpg' in pathName) or ('png' in pathName) ):
                                f.write(pathName+','+str(i)+'\n')
                    else:
                        if 'txt' in filename:
                            pathName=os.path.join(root,filename)
                            if( ('jpg' in pathName) or ('png' in pathName) ):
                                f.write(pathName+','+str(i)+'\n')
# this is v1 mapping reader
def read_x_y_mapping(mappings,basedirs,train_or_vali,shuffle,args,txt=False):
    train_mapping_file=mappings[0]
    vali_mapping_file=mappings[1]
    file_list=[]
    y=[]
    if(train_or_vali=='train'):
        mapping_file=train_mapping_file
    else:
        mapping_file=vali_mapping_file
    if(not os.path.exists(mapping_file)):
        create_x_y_mapping(mappings,basedirs,train_or_vali,txt=txt)
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

def read_mapping(mapping_file,shuffle,args,txt=False):
    file_list=[]
    y=[]
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
        fln=file_list[s0:e0+1]
        flp=file_list[s1:e1+1]
        flx=file_list[s2:e2+1]
        random.shuffle(fln)
        random.shuffle(flp)
        random.shuffle(flx)
        file_list=fln+flp+flx
    return file_list,np_utils.to_categorical(np.array(y),args.n_labels),indexes

def preprocessing_augment(data,label,args):
    data=data.resize([args.width,args.height])
    data = np.asarray(data)
    data = data.astype('float64')
    if (random.random() > 0.5 and int(label[1])==1 and args.augment==True):
        data = cv2.flip(data, 1)
    data/=255.
    return data

def yoloParser(string):
    fn=string.split(",")[0]
    y=-1
    if "negative" in fn:
        y=0
    elif "positive" in fn:
        y=1
    elif "polluted" in fn:
        y=2
    vector=np.asarray( list(map(float,string.split(",")[1].split(" "))) )
    label=np_utils.to_categorical(y,3)
    return vector,label

def vec_reader(path):
    with open(path,'r') as f:
        line=f.readline()
    return yoloParser(line)

###################################################################################

train_index=0
vali_index=0
train_indexes=[-1,-1,-1]
vali_indexes=[-1,-1,-1]
def data_generator(is_training,file_lists,y,args,indexes=None,txt=False):
    is_balanced=args.balance
    batch_size=args.batch
    if is_balanced:
        global train_indexes
        global vali_indexes
        while(1):
            file_list=[]
            label_list=[]
            if is_training:
                if train_indexes==[-1,-1,-1]:
                    train_indexes=[indexes[i][0] for i in range(3)]
                for i in range(3):
                    # flag: train_indexes[i]
                    # start: indexes[i][0]
                    # end: indexes[i][1]
                    # length: indexes[i][2]
                    for n in range(int(batch_size/3)):
                        file_list.append(file_lists[ train_indexes[i] ])
                        label_list.append(y[ train_indexes[i] ])
                        train_indexes[i]+=1
                        if(train_indexes[i]>indexes[i][1]):
                            train_indexes[i]=indexes[i][0]
            else:
                if vali_indexes==[-1,-1,-1]:
                    vali_indexes=[indexes[i][0] for i in range(3)]
                for i in range(3):
                    # flag: train_indexes[i]
                    # start: indexes[i][0]
                    # end: indexes[i][1]
                    # length: indexes[i][2]
                    for n in range(int(batch_size/3)):
                        file_list.append(file_lists[ vali_indexes[i] ])
                        label_list.append(y[ vali_indexes[i] ])
                        vali_indexes[i]+=1
                        if(vali_indexes[i]>indexes[i][1]):
                            vali_indexes[i]=indexes[i][0]
            label_list=np.asarray(label_list)
            c=list(zip(file_list,label_list))
            random.shuffle(c)
            file_list,label_list=zip(*c)
            if not txt:
                output = np.zeros([batch_size, args.height,args.width, 3])
                for i in range(batch_size):
                    output[i]=preprocessing_augment(Image.open(file_list[i]),label_list[i],args)
            else:
                output = np.zeros([batch_size, args.vector_length])
                for i in range(batch_size):
                    output[i],tmp=vec_reader(file_list[i])
            yield output, np.asarray(label_list)
            
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
            if not txt:
                output = np.zeros([batch_size, args.height,args.width, 3])
                for i in range(batch_size):
                    output[i]=preprocessing_augment(Image.open(file_list[i]),label_list[i],args)
            else:
                output = np.zeros([batch_size, args.vector_length])
                for i in range(batch_size):
                    output[i],tmp=vec_reader(file_list[i])
            yield output, label_list

###################################################################################

def load_all_valid(file_list,args,txt=False):
    if not txt:
        x_vali=np.zeros([len(file_list),args.height,args.width,args.n_labels])
        for i,f in enumerate(file_list):
            x_vali[i]=Image.open(f).resize([args.width,args.height])
        x_vali=x_vali.astype('float64')
        x_vali/=255.
        return x_vali
    else:
        vali_x = np.zeros([len(file_list), args.vector_length])
        for i,f in enumerate(file_list):
            vali_x[i],tmp= vec_reader(f)
        return vali_x
###################################################################################

def scriptBackuper(scriptName,nowtime):
    from shutil import copyfile
    newscriptName=(scriptName[:scriptName.rfind('.')]+"_"+nowtime+".py")
    copyfile('./src/'+scriptName,'./src/'+newscriptName)

####################################################################################
def __randomSD(stringLength=6):
    """Generate a random string of letters and digits """
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join(random.choice(lettersAndDigits) for i in range(stringLength))

def smote(file_list,y,args,txt=False):
    print("Performing SMOTE")
    smoteDir='./data/smote/'
    try:
        os.mkdir(smoteDir)
    except Exception as e:
        pass
    originCount=np.asarray([y.count(str(i)) for i in range(args.n_labels)])
    newTimes=[math.ceil( (np.max(originCount)/originCount[i]) )-1 for i in range(args.n_labels)]
    print("[SMOTE] Original num of data:",originCount)
    print("[SMOTE] Times to duplicate:",newTimes)
    generatedFileList=[]
    generatedLabelList=[]
    if not txt:
        print("[SMOTE] Image mode")
        try:
            os.mkdir(os.path.join(smoteDir,'img'))
        except Exception as e:
            pass
        for i,N in enumerate(newTimes):
            try:
                os.mkdir(os.path.join(smoteDir,'img',str(i)))
            except Exception as e:
                pass
            if N==0:
                continue
            print("[SMOTE] Performing on class",i)
            originFileList=file_list[y.index(str(i)):y.index(str(i))+y.count(str(i))]
            x=[]
            print("[SMOTE] Reading files")
            for file in originFileList:
                x.append(np.asarray(Image.open(file).resize([args.width,args.height])))
            x=np.asarray(x)
            x=np.reshape(x,(len(x),-1))
            k=5
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=k+1,p=2)
            print("[SMOTE] Performing KNN for k =",k)
            knn.fit(x)
            del x
            print("[SMOTE] Generating")
            for f,file in enumerate(tqdm(originFileList)):
                #print("[SMOTE] Generating...",f+1,"/",len(originFileList),end='\r')
                idx=knn.kneighbors(np.reshape(np.asarray(Image.open(file).resize([args.width,args.height])),(1,-1)), return_distance=False)
                idx=idx[0][1:]
                img=np.asarray(Image.open(file))
                img2=[np.asarray(Image.open(originFileList[id])) for id in idx]
                for n in range(N):
                    rdnFilename=__randomSD(8)
                    id=choice(range(len(img2)))
                    r=uniform(0,1)
                    newImg=((1-r)*img+r*(img2[id])).astype(int)
                    newImg= Image.fromarray(newImg.astype(np.uint8))
                    generatedFile=os.path.join(smoteDir,'img',str(i),str(r)+rdnFilename+'.jpg')
                    newImg.save(generatedFile)
                    generatedFileList.append(generatedFile)
                    generatedLabelList.append(str(i))
            print("")
    else:
        print("[SMOTE] TXT mode")
        try:
            os.mkdir(os.path.join(smoteDir,'txt'))
        except Exception as e:
            pass
        for i,N in enumerate(newTimes):
            try:
                os.mkdir(os.path.join(smoteDir,'txt',str(i)))
            except Exception as e:
                pass
            if N==0:
                continue
            print("[SMOTE] Performing on class",i)
            originFileList=file_list[y.index(str(i)):y.index(str(i))+y.count(str(i))]
            x=[]
            print("[SMOTE] Reading files")
            for file in originFileList:
                x.append(vec_reader(file)[0])
            x=np.asarray(x)
            k=5
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=k+1,p=2)
            print("[SMOTE] Performing KNN for k =",k)
            knn.fit(x)
            del x
            print("[SMOTE] Generating")
            for f,file in enumerate(tqdm(originFileList)):
                vec1=vec_reader(file)[0]
                idx=knn.kneighbors(np.reshape(vec1,(1,-1)), return_distance=False)
                idx=idx[0][1:]
                vec2=[vec_reader(originFileList[id])[0] for id in idx]
                for n in range(N):
                    rdnFilename=__randomSD(8)
                    id=choice(range(len(vec2)))
                    r=uniform(0,1)
                    newVec=((1-r)*vec1+r*(vec2[id]))
                    generatedFile=os.path.join(smoteDir,'txt',str(i),str(r)+rdnFilename+'.txt')
                    with open(generatedFile,'w') as file:
                        file.write(generatedFile+",")
                        for index,vec in enumerate(newVec):
                            file.write(str(vec))
                            if index!=len(newVec)-1:
                                file.write(" ")
                    generatedFileList.append(generatedFile)
                    generatedLabelList.append(str(i))
            print("")
    print("[SMOTE]",len(generatedFileList),"files have been generated")
    return generatedFileList+file_list,generatedLabelList+y
