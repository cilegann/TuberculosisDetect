from utils import *
import argparse
import shutil
parser=argparse.ArgumentParser(description="SMOTE generator")
parser.add_argument('--width',type=int,default=420)
parser.add_argument('--height',type=int,default=131)
parser.add_argument('--n_labels',type=int,default=3)
parser.add_argument('--data',type=str,default='190410_newdata',help='Dataset')
args=parser.parse_args()


train_mapping_file='./mapping/'+args.data+'_train_cnn_mapping.csv'
smoteDir='./data/smote/'
try:
    shutil.rmtree(smoteDir)
except Exception as e:
    pass

file_list=[]
y=[]
with open(train_mapping_file,'r') as f:
    next(f)
    lines=f.readlines()
    for line in lines:
        file_list.append(line.split(',')[0])
        y.append(line.split(',')[1][:-1])
print("Origin num of data:",len(file_list))
file_list,labels=smote(file_list,y,args)
with open('./mapping/'+args.data+"_smote_train_cnn_mapping.csv",'w') as file:
    file.write('file_path,label')
    for f,l in zip(file_list,labels):
        file.write('\n'+f+','+l)
print("Num of data:",len(file_list))
print("New mapping file has been dumped as",'./mapping/'+args.data+"_smote_train_cnn_mapping.csv")
shutil.copyfile('./mapping/'+args.data+'_vali_cnn_mapping.csv','./mapping/'+args.data+'_smote_vali_cnn_mapping.csv')

train_mapping_file='./mapping/'+args.data+'_train_yolo9000_mapping.csv'
file_list=[]
y=[]
with open(train_mapping_file,'r') as f:
    next(f)
    lines=f.readlines()
    for line in lines:
        file_list.append(line.split(',')[0])
        y.append(line.split(',')[1][:-1])
print("Origin num of data:",len(file_list))
file_list,labels=smote(file_list,y,args,txt=True)
with open('./mapping/'+args.data+"_smote_train_yolo9000_mapping.csv",'w') as file:
    file.write('file_path,label\n')
    for f,l in zip(file_list,labels):
        file.write(f+','+l+'\n')
print("Num of data:",len(file_list))
print("New mapping file has been dumped as",'./mapping/'+args.data+"_smote_train_yolo9000_mapping.csv")
shutil.copyfile('./mapping/'+args.data+'_vali_cnn_mapping.csv','./mapping/'+args.data+'_smote_vali_yolo9000_mapping.csv')