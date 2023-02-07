
from curses.ascii import isdigit
from genericpath import isdir
import os
from pathlib import Path
import shutil

data=Path("Vaihingen")
new_dir=Path("data/Vaihingen")

def for_3rd():
    test_rp="rgbs"
    test_hp="heights"
    
    test_rgbs=os.makedirs(os.path.join(new_dir,test_rp))
    test_heights=os.makedirs(os.path.join(new_dir,test_hp))
    for i in os.scandir(data):
      
        if os.path.isdir(i):
            for idx,j in enumerate(os.listdir(i)):
              a_num=""
              typ=j.split('_')[0]
              area=j.split('_')[-3]
              for k in area: 
                if k.isdigit(): a_num=a_num+k
              ij=os.path.join(Path(i),j)
              if(typ=="dsm"):
                d1=os.path.join(new_dir,test_hp)
                shutil.move(ij,d1)
                os.rename(os.path.join(d1,j),os.path.join(d1,f"ARG_{a_num}_{idx}_AGL.tif"))
              elif (typ=="top"):
                d2=os.path.join(new_dir,test_rp)
                shutil.move(ij,d2)
                os.rename(os.path.join(d2,j),os.path.join(d2,f"ARG_{a_num}_{idx}_RGB.tif"))

def modify():
    test_rp="rgbs"
    test_hp="heights"
    
    test_rgbs=os.makedirs(os.path.join(new_dir,test_rp))
    test_heights=os.makedirs(os.path.join(new_dir,test_hp))
    for i in os.scandir(data):
      
        if os.path.isdir(i):
            for idx,j in enumerate(os.listdir(i)):
              a_num=""
              typ=j.split('_')[0]
              area=j.split('_')[-3]
              for k in area: 
                if k.isdigit(): a_num=a_num+k
              ij=os.path.join(Path(i),j)
              if(typ=="dsm"):
                d1=os.path.join(new_dir,test_hp)
                shutil.move(ij,d1)
                os.rename(os.path.join(d1,j),os.path.join(d1,f"DSM_09cm_matching_{a_num}_{idx}.tif"))
              elif (typ=="top"):
                d2=os.path.join(new_dir,test_rp)
                shutil.move(ij,d2)
                os.rename(os.path.join(d2,j),os.path.join(d2,f"TOP_Mosaic_09cm_{a_num}_{idx}.tif"))
       
    
if __name__=='__main__':
    modify()
    #for_3rd()