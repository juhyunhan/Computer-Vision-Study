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
        self.files = glob.glob(path + '/camseq2007/*L.png')
        if len(self.files) == 0 :  
            raise ValueError("Dataset not found ! !")
        
        print(len(self.files))
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
        

