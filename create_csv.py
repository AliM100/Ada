from asyncio.proactor_events import _ProactorBasePipeTransport
from genericpath import isfile
import os,random
import glob 
from  pathlib import Path
from pickle import FRAME
import shutil
import csv
from tkinter import LAST
import pandas as pd

def create_from_csv():
    filename="train_Vaih.csv"
    csv_file="dataset/train_Vaihingen.csv"
    test_heights="data/Vaihingen/test_heights"
    test_rgbs="data/Vaihingen/test_rgbs"
    frame = pd.read_csv(csv_file, header=None)
    with open(filename,'w') as csvfile:
        for idx,row in frame.iterrows():
                name = frame.iloc[idx, 0]
                dsm=frame.iloc[idx,1]
                r=name.split('/')[-1]
                ds=dsm.split('/')[-1]
                for n in Path(test_rgbs).glob(r): 
                  for n in Path(test_heights).glob(ds):
                    csvwriter=csv.writer(csvfile)
                    csvwriter.writerow([name,dsm])

def create_any():
  
    filename="test_Geo.csv"
    
    if os.path.isfile(filename):
        raise Exception("file exists")
    else:
        path=Path("data/Geopose_data")
        test_rgb=os.path.join(path,"test_rgbs")
        dsm=os.path.join(path,"test_heights")
        with open(filename,'w') as csvfile:
            for i in sorted(os.listdir(test_rgb)):
              m=i.split('_')[-1]
              
              n=i.split(m)[0]
              print(n)
              for name in (Path(dsm).glob(f"{n}AGL.tif")): d=name
              print(d)
              csvwriter=csv.writer(csvfile)
              csvwriter.writerow([os.path.join(test_rgb,i),d])
              


def create_Vaihingen():
  filename="test_V.csv"
  if os.path.isfile(filename):
        raise Exception("file exists")
  else:
        path=Path("data/Vaihingen")
        test_rgb=os.path.join(path,"test_rgbs")
        dsm=os.path.join(path,"test_heights")
        with open(filename,'w') as csvfile:
            for i in sorted(os.listdir(test_rgb)):
              tif=i.split('.')[-2]
              print(tif)
              beflast=tif.split('_')[-2]
              last=tif.split('_')[-1]
              print(f"{beflast}----{last}")
              for name in (Path(dsm).glob(f"*_{beflast}_{last}.tif")): d=name
              csvwriter=csv.writer(csvfile)
              csvwriter.writerow([os.path.join(test_rgb,i),d])  



                
def split_train_test():
    filename1="train_Vaih_Custom.csv"
    filename2="test_Vaih_Custom.csv"
    heights="data/Vaihingen/heights"
    rgbs="data/Vaihingen/rgbs"
    test_heights="data/Vaihingen/test_heights"
    test_rgbs="data/Vaihingen/test_rgbs"
    train_heights="data/Vaihingen/train_heights"
    train_rgbs="data/Vaihingen/train_rgbs"
    os.mkdir(train_heights)
    os.mkdir(train_rgbs)
    os.mkdir(test_heights)
    os.mkdir(test_rgbs)
    with open(filename1,'w') as csvfile:
      for h in random.sample(os.listdir(heights),int(len(os.listdir(heights))*.8)):
        print(h)
        tif=h.split(".tif")[-2]
        print(tif)
        id1=tif.split("_")[-2]
        id2=tif.split("_")[-1]
        print(id1,id2)
        for im in Path(rgbs).glob(f"*_{id1}_{id2}.tif"): ima=im
        new_trheight=shutil.move(os.path.join(heights,h),train_heights)
        new_trrgb=shutil.move(ima,train_rgbs)
        csvwriter=csv.writer(csvfile)
        csvwriter.writerow([new_trrgb,new_trheight])  

    with open(filename2,'w') as csvfile1:
        for h in os.listdir(heights):
            print(h)
            tif=h.split(".tif")[-2]
            print(tif)
            id1=tif.split("_")[-2]
            id2=tif.split("_")[-1]
            print(id1,id2)
            for im in Path(rgbs).glob(f"*_{id1}_{id2}.tif"): ima=im
            new_teheight=shutil.move(os.path.join(heights,h),test_heights)
            new_tergb=shutil.move(ima,test_rgbs)
            csvwriter=csv.writer(csvfile1)
            csvwriter.writerow([new_tergb,new_teheight])    
      
 
def split_train_test_val_3rd():
    heights="data/Vaihingen/heights"
    rgbs="data/Vaihingen/rgbs"
    
    test="data/Vaihingen/test"
    train="data/Vaihingen/train"
    val="data/Vaihingen/valid"
    os.mkdir(train)
    os.mkdir(test)
    os.mkdir(val)
    
    for h in random.sample(os.listdir(heights),int(len(os.listdir(heights))*.7)):
        tif=h.split(".tif")[-2]
        print(tif)
        id1=tif.split("_")[-3]
        print("id1: ",id1)
        id2=tif.split("_")[-2]
        print("id2: ",id2)
        for im in Path(rgbs).glob(f"*_{id1}_{id2}_RGB.tif"): ima=im
        print(ima)
        new_trheight=shutil.move(os.path.join(heights,h),train)
        new_trrgb=shutil.move(ima,train)
    for h in random.sample(os.listdir(heights),int(len(os.listdir(heights))*.5)):
        tif=h.split(".tif")[-2]
        id1=tif.split("_")[-3]
        id2=tif.split("_")[-2]
        for im in Path(rgbs).glob(f"*_{id1}_{id2}_RGB.tif"): ima=im
        new_trheight=shutil.move(os.path.join(heights,h),test)
        new_trrgb=shutil.move(ima,test)
    for h in os.listdir(heights):
        tif=h.split(".tif")[-2]
        id1=tif.split("_")[-3]
        id2=tif.split("_")[-2]
        for im in Path(rgbs).glob(f"*_{id1}_{id2}_RGB.tif"): ima=im
        new_trheight=shutil.move(os.path.join(heights,h),val)
        new_trrgb=shutil.move(ima,val)

        

        
   
        
        
def split_train_test_val():
    heights="data/Vaihingen/heights"
    rgbs="data/Vaihingen/rgbs"
    test="data/Vaihingen/test"
    train="data/Vaihingen/train"
    val="data/Vaihingen/valid"
    filename1="train_vaih.csv"
    filename2="test_vaih.csv"
    filename3="valid_vaih.csv"

        
    # os.mkdir(train)
    # os.mkdir(test)
    # os.mkdir(val)
    # with open(filename1,'w') as csvfile:
    #     csvwriter=csv.writer(csvfile)
    #     for h in random.sample(os.listdir(heights),int(len(os.listdir(heights))*.7)):
    #         #print(h)
    #         id1=h.split("_")[-2]
    #         #print("id1: ",id1)
    #         i=h.split("_")[-1]
    #         id2=i.split(".")[0]
    #         #print("id2: ",id2)
    #         for im in Path(rgbs).glob(f"*_{id1}_{id2}.tif"): ima=im
    #         #print(ima)
    #         new_trheight=shutil.move(os.path.join(heights,h),train)
    #         new_trrgb=shutil.move(ima,train)
    #         csvwriter.writerow([new_trheight,new_trrgb])
            
    with open(filename2,'w') as csvfile:
        csvwriter=csv.writer(csvfile)         
        for h in random.sample(os.listdir(heights),int(len(os.listdir(heights))*.5)):
            id1=h.split("_")[-2]
            i=h.split("_")[-1]
            id2=i.split(".")[0]
            for im in Path(rgbs).glob(f"*_{id1}_{id2}.tif"): ima=im
            new_trheight=shutil.move(os.path.join(heights,h),test)
            new_trrgb=shutil.move(ima,test)
            csvwriter.writerow([new_trheight,new_trrgb])
        
    with open(filename3,'w') as csvfile:   
        csvwriter=csv.writer(csvfile)
        for h in os.listdir(heights):
            id1=h.split("_")[-2]
            i=h.split("_")[-1]
            id2=i.split(".")[0]
            for im in Path(rgbs).glob(f"*_{id1}_{id2}.tif"): ima=im
            new_trheight=shutil.move(os.path.join(heights,h),val)
            new_trrgb=shutil.move(ima,val)
            csvwriter.writerow([new_trheight,new_trrgb])
        
def move_all():
    # top="Vaihingen/top"
    # dsm="Vaihingen/dsm"
    # for i in os.listdir(top):
    #     shutil.move(os.path.join(top,i),"Vaihingen")
    # for i in os.listdir(dsm):
    #     shutil.move(os.path.join(dsm,i),"Vaihingen")
    train="data/Vaihingen/train"
    height="data/Vaihingen/height"
    rgbs="data/Vaihingen/rgbs"
    for i in Path(train).glob("DSM_09cm_matching_*_*.tif"):
        shutil.move(i,height)
    for j in  Path(train).glob("TOP_Mosaic_09cm_*_*.tif"):
         shutil.move(j,rgbs)
                
def copy_json():
    test="data/test"
    train="data/train"
    valid="data/valid"
    json="data/ARG_ACffgQ_VFLOW.json"
  
    for im in Path(test).glob(f"*_*_*_RGB.tif"): 
        ima=str(im)
        print(ima)
        id1=ima.split("_")[1]
        id2=ima.split("_")[2]
        print("id1: ",id1)
        print("id2: ",id2)
        path=Path(shutil.copy(json,test))
        os.rename(path,os.path.join(test,f"ARG_{id1}_{id2}_VFLOW.json"))
    for im in Path(train).glob(f"*_*_*_RGB.tif"): 
        ima=str(im)
        print(ima)
        id1=ima.split("_")[1]
        id2=ima.split("_")[2]
        print("id1: ",id1)
        print("id2: ",id2)
        path=Path(shutil.copy(json,train))
        os.rename(path,os.path.join(train,f"ARG_{id1}_{id2}_VFLOW.json"))
    for im in Path(valid).glob(f"*_*_*_RGB.tif"): 
        ima=str(im)
        print(ima)
        id1=ima.split("_")[1]
        id2=ima.split("_")[2]
        print("id1: ",id1)
        print("id2: ",id2)
        path=Path(shutil.copy(json,valid))
        os.rename(path,os.path.join(valid,f"ARG_{id1}_{id2}_VFLOW.json"))    
        
            
if __name__ == '__main__':
    
    #create_from_csv()
    #split_train_test()
    #move_all()
    split_train_test_val()
    #create_any()
    #create_Vaihingen()
    # copy_json()
    #print(len(os.listdir("data/Vaihingen/rgbs")))
          
    
              