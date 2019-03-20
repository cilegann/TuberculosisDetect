import os
import sys
modelDir='./models'
logDir='./log'
srcDir='./src'
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
ok=input("Delete? [y/n]: ")
if ok=="Y" or ok=="y":
    for f in purgeList:
        print("Deleting",f)
        os.remove(f)
    print("\nLosers are killed.\n")
                
