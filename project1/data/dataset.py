import torch
import numpy as np
from PIL import Image
from glob import glob
import random
import torchvision
import os

class LinnaeusDataset(torch.utils.data.Dataset):
    '''
    return : Ramdom combination of dog and flower images
    '''
    def __init__(self,split = 'train'):
        self.split = split
        self.base_path = os.path.dirname(os.path.abspath(__file__)) + '/Linnaeus_5_128X128/' + self.split + '/'
        
        #! dog image samples
        self.dog_path = 'dog/'
        self.dog_img_paths = self.base_path + self.dog_path
        self.dog_samples = glob(self.dog_img_paths + '*.jpg')
        
        #! flower image samples
        self.flower_path = 'flower/'
        self.flower_img_paths = self.base_path + self.flower_path
        self.flower_samples = glob(self.flower_img_paths + '*.jpg')
        
        self.transform = torchvision.transforms.ToTensor()
        
    def __len__(self):
        return len(self.dog_samples)
    
    def __getitem__(self, idx):
        main_img = Image.open(self.dog_samples[idx])
        sub_idx = np.random.randint(0,len(self.flower_samples))
        sub_img = Image.open(self.flower_samples[sub_idx])
        
        main_img = np.array(main_img)
        sub_img = np.array(sub_img)

        main_blocks = []
        sub_blocks = []

        y = 0
        for i in range(3):
            x = 0
            for j in range(3):
                main_blocks.append(main_img[y:y+40, x:x+40, :])
                sub_blocks.append(sub_img[y:y+40, x:x+40, :])
                x = x + 40
            y = y + 40
        
        
        #! 랜덤 조합 생성    
        random_idx1 = [0,1,2,3,4,5,6,7,8]
        random_idx2 = [0,1,2,3,4,5,6,7,8]

        new_img = np.full((120, 240, 3), 0, dtype=np.uint8) #8비트로 2,4,8,16,32,84,128,256

        positions = list(np.arange(0,18))
        
        labels = np.zeros(18, dtype=np.uint8)
        
        while len(positions) != 0:
            #! 1번쨰 사진 뽑기 및 위치
            random_val1 = random.choice(random_idx1)
            position = random.choice(positions)
            labels[position] = 0 #dog (main)
            
            row = position // 6
            col = position % 6 
            start_x = 40 * col
            end_x = 40 * col + 40
            
            start_y = 40 * row
            end_y = 40 * row + 40
            
            block1 = main_blocks[random_val1]
            new_img[start_y:end_y, start_x:end_x, :] = block1
            
            
            positions.remove(position)
            random_idx1.remove(random_val1)
            
            #! 2번째 사진 뽑기 및 위치 시키기
            random_val2 = random.choice(random_idx2)
            position = random.choice(positions)
            labels[position] = 1 #flower (sub)
            row = position // 6
            col = position % 6 
            start_x = 40 * col
            end_x = 40 * col + 40
            start_y = 40 * row
            end_y = 40 * row + 40
            

            block2 = sub_blocks[random_val2]
            
            
            new_img[start_y:end_y, start_x:end_x, :] = block2
            
            positions.remove(position)
            random_idx2.remove(random_val2)
             
        img = self.transform(new_img)
        labels = torch.tensor(labels)
        
        return img, labels