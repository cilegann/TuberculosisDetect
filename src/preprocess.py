import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
import sys

negFilePrefix='./data/negative/negative_'
polFilePrefix='./data/polluted/polluted_'
posFilePrefix='./data/positive/positive_'
Files={negFilePrefix,polFilePrefix,posFilePrefix}
IMAGE_MAX_NUM=20

def readImage(neg,pol,pos):
    data=[]
    for f in Files:
        for i in range(1,IMAGE_MAX_NUM+1):
            filename=f+str(i)+".jpg"
            print("   Reading "+filename)
            im=io.imread(filename)
            data.append(im.tolist())
    data=np.array(data)
    filename='./'+str(neg)+'-'+str(pol)+'-'+str(pos)+'.npy'
    np.save(filename,data)
    print("   File saved as "+filename)
    return data
def getData(neg,pol,pos):
    filename='./'+str(neg)+'-'+str(pol)+'-'+str(pos)+'.npy'
    if os.path.exists(filename):
        print("   Load images from "+filename)
        data=np.load(filename)
    else:
        data=readImage(neg,pol,pos)
    return data

def getMean(data):
    if '-recompute' in sys.argv or not(os.path.exists(meanF)):
        print("   Calculating mean")
        data_m=data.mean(axis=0,keepdims=True)
        print(data_m.shape())
        # shape=(1,600,600,3)
        data_m=data_m.reshape(600,600,3)
        print(data_m.shape())
        # shape=(600,600,3)
        np.save(meanF,data_m)
        print("   Mean caculated and saved as "+meanF)
        
    else:
        print("   Load mean form "+meanF)
        data_m=np.load(meanF)
    return data_m

data=getData(20,20,20).reshape(60,1678*524*3)
data_m=getMean(data).astype(int)