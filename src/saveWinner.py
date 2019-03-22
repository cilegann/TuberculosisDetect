import os
import sys
from shutil import copyfile,move
modelDir='./models'
modelSaveDir='./saved_do_not_del_models'
srcDir='./src'
srcSaveDir='./saved_do_not_del_src'
print("\n\n")
print(" __     __               _                      _   _ _  \n\
 \\ \\   / /              | |                    | | (_) |  \n\
  \\ \\_/ /__  _   _   ___| | __ _ _   _  ___  __| |  _| |_ \n\
   \\   / _ \\| | | | / __| |/ _` | | | |/ _ \\/ _` | | | __|\n\
    | | (_) | |_| | \\__ \\ | (_| | |_| |  __/ (_| | | | |_ \n\
    |_|\\___/ \\__,_| |___/_|\\__,_|\\__, |\\___|\\__,_| |_|\\__|\n\
                                  __/ |                   \n\
                                 |___/                    \n\
")
print("                    _______________ \n\
                   |@@@@|     |####|\n\
                   |@@@@|     |####|\n\
                   |@@@@|     |####|\n\
                   \@@@@|     |####/\n\
                    \@@@|     |###/\n\
                     `@@|_____|##'\n\
                          (O)\n\
                       .-'''''-.\n\
                     .'  * * *  `.\n\
                    :  *       *  :\n\
                   : ~ WELL DONE ~ :\n\
                   : ~  SAVE IT  ~ :\n\
                    :  *       *  :\n\
                     `.  * * *  .'\n\
                       `-.....-'\n\
")
for d,s in zip([modelDir],[modelSaveDir]):
    for r,ds,fs in os.walk(d):
        for f in fs:
            ok=False
            for a in sys.argv[1:]:
                if a in f:
                    ok=True
                    break
            if ok:
                print("Copying:",os.path.join(d,f),"->",os.path.join(s,f))
                copyfile(os.path.join(d,f),os.path.join(s,f))

for d,s in zip([srcDir],[srcSaveDir]):
    for r,ds,fs in os.walk(d):
        for f in fs:
            ok=False
            for a in sys.argv[1:]:
                if a in f:
                    ok=True
                    break
            if ok:
                print("Moving:",os.path.join(d,f),"->",os.path.join(s,f))
                move(os.path.join(d,f),os.path.join(s,f))

                
