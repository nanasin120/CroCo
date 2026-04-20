import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_loss(predict, target, isMasked):
    # predict : [B, 196, 768]
    # target : [B, 3, 224, 224]
    p = 16
    B, C, H, W = target.shape
    target = target.reshape(B, C, H//p, p, W//p, p) # [B, 3, 224/16, 224/16, 16]
    target = target.permute(0, 2, 4, 3, 5, 1).reshape(B, (H//p)*(W//p), -1) # [B, 196, 768]

    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True, unbiased=False)
    target = (target-mean) / (var + 1e-6)**0.5

    loss = (predict - target) ** 2 # [B, 196, 768]
    loss = loss.mean(dim=-1) # [B, 196]

    loss = (loss * isMasked).sum() / (isMasked.sum() + 1e-6)

    return loss