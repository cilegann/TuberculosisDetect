import os
import shutil
import numpy as np
from vis.visualization import visualize_saliency, visualize_cam,overlay
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from texttable import Texttable
import itertools

def cam(filename,img,label,model,backprop_modifier='guided'):
    #shutil.rmtree("./cam/")
    #os.mkdir("cam")
    np.seterr(divide='ignore',invalid='ignore')
    
    filename='./cam/'+filename+'.jpg'
    print(filename)
    heatmap = visualize_cam(model, layer_idx=-1, filter_indices=label, seed_input=img,backprop_modifier=backprop_modifier)
    jet_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    im1=plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    im2 = plt.imshow(heatmap,  alpha=.4, interpolation='bilinear')
    plt.savefig(filename,dpi=300)
    plt.close()


def plot_confusion_matrix_by_cm(cmx,classes,filename='cm',title='Confusion matrix',cmap=plt.cm.Blues,save=True):
    cmx=cmx.astype('float')/cmx.sum(axis=1)[:,np.newaxis]
    print("\n CONFUSION MATRIX")
    print(cmx)
    #plt.show()
    plt.figure()
    plt.imshow(cmx,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    plt.tight_layout()
    plt.ylabel("Real")
    plt.xlabel("Predict")
    for i, j in itertools.product(range(cmx.shape[0]), range(cmx.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cmx[i, j]), horizontalalignment="center",
            color="white" if cmx[i, j] > 0.5 else "black",fontsize=20) 
    if save: 
        if not os.path.exist("./confusionMatrix"):
            os.path.mkdir("./confusionMatrix")
        plt.savefig('./confusionMatrix/'+filename+'.png')
    try:
        plt.show()
    except:
        pass
    plt.close()

def plot_confusion_matrix(y_true,y_pred,classes,filename='confusion_matrix',title='Confusion matrix',cmap=plt.cm.Blues):
    labels=["negative", "positive", "polluted"]
    cmx = confusion_matrix(y_true,y_pred)
    cmx=cmx.astype('float')/cmx.sum(axis=1)[:,np.newaxis]
    plot_confusion_matrix_by_cm(cmx,classes,filename,title,cmap)
    

def evaluate(y_true,y_pred):
    precision, recall, fscore, support = score(y_true, y_pred)
    t=Texttable()
    content=[['Label','Precision','Recall','FScore','Support']]
    for i in range(len(precision)):
        c=[str(i),precision[i],recall[i],fscore[i],support[i]]
        content.append(c)
    t.add_rows(content)
    print("\nSCOREs\n"+t.draw()+"\n")
    pass

