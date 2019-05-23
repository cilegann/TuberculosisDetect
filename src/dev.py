import argparse
from utils import *
parser=argparse.ArgumentParser(description="CNN on TB")
parser.add_argument('--tstrain',action='store_true',help='Training mode (positove first)')
parser.add_argument('--train',action='store_true',help='Training mode')
parser.add_argument('--test',action='store_true',help='Testing mode')
parser.add_argument('--dev',action='store_true',help='Dev mode')
parser.add_argument('-m','--model',type=str,help='The model you want to test on')
parser.add_argument('--best',action='store_true',help='Load best model or not')
parser.add_argument('--width',type=int,default=420)
parser.add_argument('--height',type=int,default=131)
parser.add_argument('--batch',type=int,default=32,help='Batch size')
parser.add_argument('--epochs',type=int,default=200,help='#Epochs')
parser.add_argument('--balance',action='store_true',help='Balance data by undersampling the majiroty data')
parser.add_argument('--n_labels',type=int,default=3)
parser.add_argument('--gpu',type=str,default='1',help='No. of GPU to use')
parser.add_argument('--data',type=str,default='190410_newdata',help='Dataset')
parser.add_argument('--augment',action='store_true',help='Data augment by randomly flipping image')
args=parser.parse_args()
    
train_mapping_file='./mapping/'+args.data+'_train_cnn_mapping.csv'
vali_mapping_file='./mapping/'+args.data+'_vali_cnn_mapping.csv'
args.mappings=[train_mapping_file,vali_mapping_file]
mapping_file=args.mappings[0]

file_list=[]
y=[]
with open(mapping_file,'r') as f:
    next(f)
    lines=f.readlines()
    for line in lines:
        file_list.append(line.split(',')[0])
        y.append(line.split(',')[1][:-1])
print(len(file_list))
a,b=smote(file_list,y,args)
print(len(a))
