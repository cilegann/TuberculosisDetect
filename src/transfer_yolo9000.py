import numpy as np
from keras import utils as np_utils

train_vector_file='./shuffled_train_vector.txt'
vali_vector_file='./shuffled_vali_vector.txt'

def parser(string):
    fn=string.split(",")[0]
    if "negative" in fn:
        y=0
    elif "positive" in fn:
        y=1
    elif "polluted" in fn:
        y=2
    vector=np.asarray( list(map(float,string.split(",")[1].split(" "))) )
    label=np_utils.to_categorical(y,3)
    return vector,label
    
def data_generator(train_vali):
    index=0
    if(train_vali=='train'):
        filename=train_vector_file
    else:
        filename=vali_vector_file
    while(1):
        with open(filename,'r') as file:

