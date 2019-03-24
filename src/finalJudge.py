import os
from keras.models import load_model
from PIL import Image
import numpy as np
from shutil import copyfile

model_file='./tscnn.h5'
path='./fileNameOK'
categ='./machine_categ'


model=load_model(model_file)
if not os.path.exists(os.path.join(categ,'n')):
    os.mkdir(os.path.join(categ,'n'))
if not os.path.exists(os.path.join(categ,'p')):
    os.mkdir(os.path.join(categ,'p'))
if not os.path.exists(os.path.join(categ,'x')):
    os.mkdir(os.path.join(categ,'x'))

def judge(orgPath,f):
    img=Image.open(orgPath).resize([420,131])
    img=np.asarray(img).astype('float64')
    img/=255.
    img=np.reshape(img,[1,131,420,3])
    result=np.argmax(model.predict(img))
    print(orgPath,result)
    if result==1:
        copyfile(orgPath,os.path.join(os.path.join(categ,'p'),f))
    elif result==2:
        copyfile(orgPath,os.path.join(os.path.join(categ,'x'),f))
    elif result==0:
        copyfile(orgPath,os.path.join(os.path.join(categ,'n'),f))

for r,ds,fs in os.walk(path):
    for f in fs:
        if '.jpg' in f:
            orgPath=os.path.join(r,f)
            judge(orgPath,f)