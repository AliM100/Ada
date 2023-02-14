from genericpath import isdir
import os
from pathlib import Path
from glob import glob
from pip import main
import re
import shutil
data=Path("geo_data")


def check(type):
    a=list(data.glob(f"*_*_{type}.*"))
    print(len(a))
    
def move():
   n=0
   for i in os.listdir(data):
        temp=i.split("_")[2]
        y=re.split(f"_{temp}",i)[0]
        print(y)
         
        if(list(data.glob(f"{y}_AGL.tif")) and list(data.glob(f"{y}_RGB.tif"))): #and list(data.glob(f"{y}_VFLOW.json"))):
            continue 
        else:
            print("move")
            shutil.move(os.path.join(data,i),new)
 
           
   


if __name__ == "__main__":
   #remove folder
    # location="delete"
    # shutil.rmtree(location)
    new=Path("missed_train_data")
    if os.path.isdir(new):
        pass
    else: 
      os.mkdir(new)
    #check("RGB") 
    move()
   
            
         
     
         
     