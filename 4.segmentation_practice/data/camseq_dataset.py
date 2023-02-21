import torch
import numpy as np
import glob
from torch.utils import data
from PIL import Image
import os

class CamSeq(data.Dataset): 
    def __init__(self):
        super().__init__()
        path = os.path.dirname(os.path.abspath(__file__))
        print(path)
        self.files = glob.glob(path + '/camseq2007/*L.png') #리스트에 담김
        if len(self.files) == 0 :  
            raise ValueError("Dataset not found ! !")
        
        print(len(self.files))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = self.files[index]
        label = Image.open(label_path)
        
        label = Image.open(self.files[index])
        img_path = img_path[:-6] + '.png'

