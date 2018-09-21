# -*- coding: utf-8 -*-
import os
basedir='D:/NTU/MS/TB/original_data/categ/Q/'
for fn in os.listdir(basedir):
    print(fn)
    print(os.path.join(basedir,fn))
    '''
    if('.jpg' in fn):
        id=fn[:fn.find('_')]
        folderPath=os.path.join(basedir,fn[:fn.find('_')])
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        else:
            os.rename(os.path.join(basedir,fn),os.path.join(folderPath,fn)+'/')
    '''