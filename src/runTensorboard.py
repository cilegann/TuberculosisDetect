import os
import sys
time=sys.argv[1].replace("-","").replace(":","")
p=""
for r,ds,fs in os.walk("./Graph"):
    for d in ds:
        if time in d:
            p=os.path.join(r,d)
            break
os.system("tensorboard --logdir='"+p+"'")