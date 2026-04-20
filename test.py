import torch
import torch.nn as nn
import torch.nn.functional as F
from CroCO import CroCO

# image1 : [B, 3, H, W] [B, 3, 224, 224]
# image2 : [B, 3, H, W] [B, 3, 224, 224]

B, C, H, W = 8, 3, 224, 224

model = CroCO()

dummy_image1 = torch.randn(B, C, H, W)
dummy_image2 = torch.randn(B, C, H, W)

p1, _ = model(dummy_image1, dummy_image2)

print(p1.shape)