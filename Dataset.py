import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class CroCoDataset(Dataset):
    def __init__(self, img_dir, frame_interval=15):
        self.img_dir = img_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.frame_interval = frame_interval

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # 혹시 모르니 리사이즈 추가
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.img_files) - self.frame_interval # 2개씩 하니 맨뒤 날림
    
    def __getitem__(self, idx):
        src1_path = os.path.join(self.img_dir, self.img_files[idx])
        src2_path = os.path.join(self.img_dir, self.img_files[idx + self.frame_interval])

        img1 = self.transform(Image.open(src1_path).convert('RGB'))
        img2 = self.transform(Image.open(src2_path).convert('RGB'))

        return {
            'image1' : img1,
            'image2' : img2
        }