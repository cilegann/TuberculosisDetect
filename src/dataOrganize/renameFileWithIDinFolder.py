import os
RENAME=0
#os.chdir( os.path.dirname(os.path.abspath(__file__)) )
emptyList=[]
validList=[]
totalImage=0
for root, directories, filenames in os.walk('./original_data/fileNameOK/'):
    #print("ROOT: "+root)
    No=root[root.rfind("/")+1:]
    print("NO  : "+No)
    photo=0
    for filename in filenames:
        #print(" > FLNM: "+filename)
        if('.jpg' in filename):
            photo+=1
            totalImage+=1
            if(RENAME==1):
                oldName=root+"/"+filename
                newName=root+"/"+No+'_'+filename
                print("   > Chage from "+oldName+" to "+newName)
                os.rename(oldName,newName)
    if photo==0:
        emptyList.append(No)
    else:
        validList.append(No)       
    print(" > "+str(photo)+" image(s) found.\n")
print("\nTotal: "+str(totalImage)+" images in "+str(len(validList))+" folders. "+str(len(emptyList))+" empty folders.")