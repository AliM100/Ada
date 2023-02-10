from distutils.command.config import config
import json
import os
from glob import glob
from pathlib import Path
from osgeo import gdal
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm


from functools import partial
import torch.optim as optim 
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


RNG = np.random.RandomState(4321)
UNITS_PER_METER_CONVERSION_FACTORS = {"cm": 100.0, "m": 1.0}

def load_image(
    image_path,
    args,
    dtype_out="float32",
    units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
):

    image_path = Path(image_path)
    if not image_path.exists():
        return None
    image = gdal.Open(str(image_path))
    print("path is :",image_path)
    
    image = image.ReadAsArray()
    # convert AGL units and fill nan placeholder with nan
    if "DSM" in image_path.name:
        ##image = image.astype(dtype_out)
       
        image1=Image.open(str(image_path))
    
        
        image = torch.from_numpy(np.array(image1))
        nchannel = len(image1.mode)
        image = image.view(image1.size[1], image1.size[0],nchannel)
       
        ##image=np.asarray(Image.open(image_path), dtype=np.float32)
        ##np.putmask(image, image == args.nan_placeholder, np.nan)
        
        units_per_meter = units_per_meter_conversion_factors[args.unit]
        image = np.array((image / units_per_meter),dtype=dtype_out)
        image=np.rollaxis(image,2,0)
        ##print("------",image.shape)
        ##image = (image / units_per_meter).astype(dtype_out)
        
    #transpose if RGB
    if len(image.shape) == 3:
        image = np.transpose(image, [1, 2, 0])
    # if len(image.getbands()) == 3:
    #         image = np.transpose(image, [1, 2, 0])
       

    return image




basemodel_name ="timm-regnety_032"
class Dataset(BaseDataset):
    is_test=False
    def __init__(self, sub_dir, args, rng=RNG, crop_size=456):
        
        self.is_test = sub_dir == args.test_data
        if self.is_test:
            print("sub_dir test")
        self.rng = rng
        self.crop_size = crop_size
       
        # create all paths with respect to RGB path ordering to maintain alignment of samples
        dataset_dir = Path(sub_dir) 
        
        rgb_paths = list(dataset_dir.glob(f"TOP_Mosaic_09cm_*_*.tif"))

        if rgb_paths == []:
            rgb_paths = list(
                dataset_dir.glob(f"TOP_Mosaic_09cm_*_*.tif")
            )  # original file names
        agl_paths = list(dataset_dir.glob(f"DSM_09cm_matching_*_*.tif"))
       

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

            self.paths_list = [
                self.paths_list[ind]
                for ind in self.rng.permutation(len(self.paths_list))
            ]
            # if args.sample_size is not None:
            #     self.paths_list = self.paths_list[: args.sample_size]
          
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
           basemodel_name, "imagenet"
        )

        self.args = args
        self.sub_dir = sub_dir

    def __getitem__(self, i):
        
        if self.is_test:
            rgb_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
        else:
            rgb_path, agl_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
            agl = load_image(agl_path, self.args)
            

            # v6_random_crop
            if self.args.random_crop:
                crop_size = self.crop_size
                #**x0 = np.random.randint(2048 - crop_size)
                #**y0 = np.random.randint(2048 - crop_size)
                x0 = np.random.randint(500 - crop_size)
                y0 = np.random.randint(500 - crop_size)
                # print(image.shape, agl.shape, mag.shape, flush=True)
                image = image[x0 : x0 + crop_size, y0 : y0 + crop_size]
                agl = agl[x0 : x0 + crop_size, y0 : y0 + crop_size]
                ##mag = mag[x0 : x0 + crop_size, y0 : y0 + crop_size]
                # print(image.shape, agl.shape, mag.shape, flush=True)

          
            # if self.args.augmentation:
            #     image, mag, xdir, ydir, agl, scale = augment_vflow(
            #         image,
            #         mag,
            #         xdir,
            #         ydir,
            #         vflow_data["angle"],
            #         vflow_data["scale"],
            #         agl=agl,
            #     )
            agl=np.rollaxis(agl,2,0)
            agl = agl.astype("float32")
          
        if self.is_test and self.args.downsample > 1:
            image = cv2.resize(
                image,
                (
                    int(image.shape[0] / self.args.downsample),
                    int(image.shape[1] / self.args.downsample),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            crop_size = self.crop_size
            x0 = np.random.randint(500 - crop_size)
            y0 = np.random.randint(500 - crop_size)
            image = image[x0 : x0 + crop_size, y0 : y0 + crop_size]
            
            
        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))
    
        ##print("agl shape before return",agl.shape)
        if self.is_test:
            return image, str(rgb_path)
        else:
            return image, agl, str(rgb_path)

    def __len__(self):
        return len(self.paths_list)
