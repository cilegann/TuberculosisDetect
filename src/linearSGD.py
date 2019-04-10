import os
import time
import random
import numpy as np
import cv2
from PIL import Image
from evaluate_tools import plot_confusion_matrix,evaluate
from utils import *
from sklearn import linear_model
import pickle
from tqdm import tqdm

def get_model(args):
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3,loss='log')
    return model

def train(args):
    model=get_model(args)
    
    # if not os.path.exists('./log'):
    #     os.mkdir('./log')
    nowtime=time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    print("######### TRAINING FILE POSTFIX #########")
    print(" "*13,nowtime)
    print(" "*13,nowtime.replace("-","").replace(":",""))
    print("#########################################")
    scriptBackuper(os.path.basename(__file__),nowtime)

    x_train_list,y_train,indexes=read_mapping(args.mappings[0],not args.balance,args)
    x_vali_list,y_vali,_=read_mapping(args.mappings[1],False,args)
    x_vali=load_all_valid(x_vali_list,args)
    #y_vali=np.argmax(y_vali,axis=1)
    #y_vali=y_vali.reshape((len(y_vali)))
    x_vali=x_vali.reshape((len(x_vali),-1))
    if args.balance:
        steps=min(np.asarray([indexes[i][2] for i in range(3)]))//(args.batch//3)
    else:
        steps=int(len(x_train_list))//int(args.batch)
    try:
       #fit
        for epo in range(args.epoch):
            print("Epoch",epo)
            for step in tqdm(range(steps)):
                x,y=next(data_generator(True,x_train_list,y_train,args,indexes))
                n_samples = len(x)
                x = x.reshape((n_samples, -1))
                y=np.argmax(y,axis=1)
                y = y.reshape((n_samples))
                model.partial_fit(x,y,classes=np.asarray([0,1,2]))
        y_pred=model.predict(x_vali)
        #y_pred=np.argmax(y_pred,axis=1)
        y_pred=y_pred.reshape((len(y_pred),1))
        y_ture=np.argmax(y_vali,axis=1)
        labels=['negative','positive','polluted']
        plot_confusion_matrix(y_ture,y_pred,labels)
        evaluate(y_ture,y_pred)
    except KeyboardInterrupt:
        with open('models/linearSGD_'+nowtime+".pickle",'wb') as file:
            pickle.dump(model,file)
        os.system("sh purge.sh "+nowtime)
    with open('models/linearSGD_'+nowtime+".pickle",'wb') as file:
        pickle.dump(model,file)

def test(args):
    with open(args.model,'rb') as file:
        model=pickle.load(file)
    x_vali_list,y_vali,_=read_mapping(args.mappings[1],False,args)
    x_vali=load_all_valid(x_vali_list,args)
    x_vali=x_vali.reshape((len(x_vali),-1))
    y_pred=model.predict(x_vali)
    y_pred=y_pred.reshape((len(y_pred),1))
    y_ture=np.argmax(y_vali,axis=1)
    labels=['negative','positive','polluted']
    plot_confusion_matrix(y_ture,y_pred,labels)
    evaluate(y_ture,y_pred)

def dev(args):
    model=get_model(args)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Two stage cnn on TB")
    parser.add_argument('--train',action='store_true',help='Training mode')
    parser.add_argument('--test',action='store_true',help='Testing mode')
    parser.add_argument('--dev',action='store_true',help='Dev mode')
    parser.add_argument('-m','--model',type=str,help='The model you want to test on')
    parser.add_argument('--width',type=int,default=420)
    parser.add_argument('--height',type=int,default=131)
    parser.add_argument('--batch',type=int,default=32,help='Batch size')
    parser.add_argument('--epoch',type=int,default=200,help='Epochs')
    parser.add_argument('--balance',action='store_true',help='Balance data by undersampling the majiroty data')
    parser.add_argument('--n_labels',type=int,default=3)
    parser.add_argument('--data',type=str,default='190408_newdata',help='Dataset')
    parser.add_argument('--augment',action='store_true')
    args=parser.parse_args()

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
