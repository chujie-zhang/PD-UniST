import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset_MIST_prompt(Dataset):
    def __init__(self, root, count=None, transforms_1=None, transforms_2=None, transforms_3=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.transform3 = transforms.Compose(transforms_3)
        self.train_transform = transforms.Compose(transforms_2)
        
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        #self.mask_A = sorted(glob.glob("%s/mask_A/*" % root))
        #self.mask_B = sorted(glob.glob("%s/mask_B/*" % root))
        self.unaligned = unaligned
        
    def __getitem__(self, index):
	    

        A_path = self.files_A[index % len(self.files_A)]
        B_path = self.files_B[index % len(self.files_B)]
  
        label_A = A_path.split('_')[-1]
        labelA = int(label_A.split('.')[0])
        label_B = B_path.split('_')[-1]
        labelB = int(label_B.split('.')[0])

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        #A_img = Image.open(A_path).convert('L')
        #B_img = Image.open(B_path).convert('L')

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        item_A = self.transform1(A_img)
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        item_B = self.transform1(B_img)
        
        return {'A': item_A, 'B': item_B,'A_label': labelA, 'B_label': labelB}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ValDataset_MIST_prompt(Dataset):
    def __init__(self, root,count = None,transforms_1=None, transforms_2=None, transforms_3=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.transform3 = transforms.Compose(transforms_3)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        
        A_path = self.files_A[index % len(self.files_A)]
        B_path = self.files_B[index % len(self.files_B)]
        
        label_A = A_path.split('_')[-1]
        labelA = int(label_A.split('.')[0])
        label_B = B_path.split('_')[-1]
        labelB = int(label_B.split('.')[0])
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        item_A = self.transform1(A_img)
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        item_B = self.transform1(B_img)
        
        return {'A': item_A, 'B': item_B,'A_label': labelA, 'B_label': labelB}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    

    
