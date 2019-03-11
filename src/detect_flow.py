#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import os
from evaluate_tools import evaluate,plot_confusion_matrix
cnn_prob_file='./result/cnn_prob.csv'
tscnn_prob_file='./result/tscnn_prob.csv'
yolo_prob_file='./result/yolo_prob.csv'
tsyolo_prob_file='./result/tsyolo_prob.csv'


# In[4]:


def read_probs():
    probs={}
    
    with open(cnn_prob_file) as f:
        next(f)
        lines=f.readlines()
    for line in lines:
        data=line.replace("\n","").split(",")
        fn=data[0].split("/")[4].replace(".txt","").replace(".jpg","")
        if fn not in probs:
            probs[fn]={}
            probs[fn]["real"]=float(data[1])
        probs[fn]["cnn"]={}
        probs[fn]["cnn"]["prob"]=np.asarray([float(i) for i in data[2:5]])
        probs[fn]["cnn"]["result"]=np.argmax(probs[fn]["cnn"]["prob"])
            
    with open(tscnn_prob_file) as f:
        next(f)
        lines=f.readlines()
    for line in lines:
        data=line.replace("\n","").split(",")
        fn=data[0].split("/")[4].replace(".txt","").replace(".jpg","")
        if fn not in probs:
            probs[fn]={}
            probs[fn]["real"]=float(data[1])
        probs[fn]["tscnn"]={}
        probs[fn]["tscnn"]["prob"]=np.asarray([float(i) for i in data[2:5]])
        probs[fn]["tscnn"]["result"]=np.argmax(probs[fn]["tscnn"]["prob"])
    
    with open(yolo_prob_file) as f:
        next(f)
        lines=f.readlines()
    for line in lines:
        data=line.replace("\n","").split(",")
        fn=data[0].split("/")[4].replace(".txt","").replace(".jpg","")
        if fn not in probs:
            probs[fn]={}
            probs[fn]["real"]=float(data[1])
        probs[fn]["yolo"]={}
        probs[fn]["yolo"]["prob"]=np.asarray([float(i) for i in data[2:5]])
        probs[fn]["yolo"]["result"]=np.argmax(probs[fn]["yolo"]["prob"])
            
    with open(tsyolo_prob_file) as f:
        next(f)
        lines=f.readlines()
    for line in lines:
        data=line.replace("\n","").split(",")
        fn=data[0].split("/")[4].replace(".txt","").replace(".jpg","")
        if fn not in probs:
            probs[fn]={}
            probs[fn]["real"]=float(data[1])
        probs[fn]["tsyolo"]={}
        probs[fn]["tsyolo"]["prob"]=np.asarray([float(i) for i in data[2:5]])
        probs[fn]["tsyolo"]["result"]=np.argmax(probs[fn]["tsyolo"]["prob"])
    
    return probs


# In[5]:


probs=read_probs()


# In[10]:


def voting_a(probs):
    true=[]
    pred=[]
    for k in sorted(probs,key=lambda k:probs[k]["real"]):
        true_value=probs[k]["real"]
        pred_value=[probs[k]["cnn"]["result"],probs[k]["tscnn"]["result"],probs[k]["yolo"]["result"],probs[k]["tsyolo"]["result"]]
        true.append(true_value)
        if pred_value.count(1)>=2:
            pred.append(1)
        elif pred_value.count(2)>=3:
            pred.append(2)
        else:
            pred.append(0)
    return np.asarray(true),np.asarray(pred)
t,p=voting_a(probs)
evaluate(t,p)
plot_confusion_matrix(t,p,["n","p","x"])


# In[12]:


def voting_b(probs):
    true=[]
    pred=[]
    for k in sorted(probs,key=lambda k:probs[k]["real"]):
        true_value=probs[k]["real"]
        pred_value=[probs[k]["cnn"]["result"],probs[k]["tscnn"]["result"],probs[k]["yolo"]["result"],probs[k]["tsyolo"]["result"]]
        true.append(true_value)
        if pred_value.count(1)>=2:
            pred.append(1)
        elif pred_value.count(2)>=2:
            pred.append(2)
        else:
            pred.append(0)
    return np.asarray(true),np.asarray(pred)
t,p=voting_b(probs)
evaluate(t,p)
plot_confusion_matrix(t,p,["n","p","x"])


# In[ ]:




