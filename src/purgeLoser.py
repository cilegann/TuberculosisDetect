import os
import sys
import shutil
modelDir='./models'
logDir='./log'
srcDir='./src'
graphDir='./Graph'
purgeList=[]
print("\n\n")
print('     _.--""--._')
print('    /  _    _  \ ')
print(' _  ( (_\  /_) )  _')
print('{ \._\   /\   /_./ }')
print('/_"=-.}______{.-="_\ ')
print(' _  _.=("""")=._  _')
print('(_\'"_.-"`~~`"-._"\'_)')
print(' {_"            "_}')
print("    LOSER PURGER\n")
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
                
