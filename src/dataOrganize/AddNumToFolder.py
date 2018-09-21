import os
basedir='D:/NTU/MS/TB/original_data/fileNameOK/'
n=1
for fn in os.listdir(basedir):
    if os.path.isdir(os.path.join(basedir,fn)):
        os.rename(os.path.join(basedir,fn),os.path.join(basedir,str(n)+'-'+fn))
        n+=1