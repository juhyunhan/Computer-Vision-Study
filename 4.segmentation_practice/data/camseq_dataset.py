import torch
import numpy as np
import glob
from torch.utils import data
import torchvision
from PIL import Image
import os

class CamSeqDatset(data.Dataset): 
    def __init__(self):
        super().__init__()
        path = os.path.dirname(os.path.abspath(__file__))
        self.files = glob.glob(path + '/camseq2007/*label.png') #리스트에 담김
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize((144,192))])
        if len(self.files) == 0 :  
            raise ValueError("Dataset not found ! !")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        label_path = self.files[index]
        label = Image.open(label_path)
        label = np.array(label)
        label = self.transform(label)
        
        img_path = label_path[:-10] + '.png'
        img = Image.open(img_path)
        img = np.array(img) # H, W, C 0-255사이
        img = self.transform(img) #C , H , W 0-1사이 (다 255로 나눈다 -> input은 nomalization이 되어있어야 하기 때문에, 안하면 gradient가 폭팔함)

        return img, label
        #data augmentation 도 있다~ 좌우 뒤집고 크랍하고 등등.. 근데 여기선 안한당