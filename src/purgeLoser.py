import os
import sys
import shutil
from art import *
modelDir='./models'
logDir='./log'
srcDir='./src'
graphDir='./Graph'
purgeList=[]
tprint("Purge Loser")
print(" "*10+'                _.--""--._')
print(" "*10+'               /  _    _  \ ')
print(" "*10+'            _  ( (_\  /_) )  _')
print(" "*10+'           { \._\   /\   /_./ }')
print(" "*10+'           /_"=-.}______{.-="_\ ')
print(" "*10+'            _  _.=("""")=._  _')
print(" "*10+'           (_\'"_.-"`~~`"-._"\'_)')
print(" "*10+'            {_"            "_}\n')
for d in [modelDir,logDir,srcDir]:
    for r,ds,fs in os.walk(d):
        for f in fs:
            ok=False
            for a in sys.argv[1:]:
                if a in f:
                    ok=True
                    break
            if ok:
                purgeList.append(os.path.join(r,f))
                print(os.path.join(r,f))
for r,ds,fs in os.walk(graphDir):
    for d in ds:
        ok=False
        for a in sys.argv[1:]:
            if a.replace("-","").replace(":","") in d:
                ok=True
                break
        if ok:
            purgeList.append(os.path.join(r,d))
            print(os.path.join(r,d))

ok=input("Delete? [y/n]: ")
if ok=="Y" or ok=="y":
    for f in purgeList:
        print("Deleting",f)
        if "Graph" not in f:
            os.remove(f)
        else:
            shutil.rmtree(f)
    print("\nLosers are killed.\n")
                
