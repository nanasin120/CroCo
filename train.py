import torch
import torch.nn as nn
import torch.nn.functional as F
from Dataset import CroCoDataset
from torch.utils.data import Subset, DataLoader
from loss import cross_loss
from CroCO import CroCO
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def to_imshow(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy() # [H, W, C]
    img = np.clip(img, 0, 1)
    return img

def logit2image(logit, target):
    # target : [3, H, W]
    p = 16 
    h = w = 14 

    target = target.to(logit.device)

    target = target.reshape(3, h, p, w, p) # [3, h, p, w, p]
    target = target.permute(1, 3, 2, 4, 0)
    target = target.reshape(h * w, p * p * 3)

    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True, unbiased=False)

    x = logit * (var + 1e-6)**0.5 + mean

    x = x.reshape(h, w, p, p, 3) # [196, 768] [14, 14, 16, 16, 3]
    x = x.permute(4, 0, 2, 1, 3).reshape(3, h*p, w*p)

    x = torch.clip(x, 0, 1)

    return x

def image2masked(image, isMasked):
    p = 16 
    h = w = 14 

    m = isMasked.reshape(h, w) # [14, 14]

    m = m.unsqueeze(-1).unsqueeze(-1).expand(h, w, p, p) # [14, 14, 16, 16]

    m = m.permute(0, 2, 1, 3).reshape(h*p, w*p).to(image.device) # [224, 224]

    img_masked = image * (1 - m.unsqueeze(0))

    return img_masked

model_save_path = r'./model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Batch = 8
learning_rate = 5e-5
warmup_epoch = 40
Epoch = 400
save_interval = 10

img_dir = r'data\cup'
full_dataset = CroCoDataset(img_dir=img_dir, frame_interval=15)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset = Subset(full_dataset, range(train_size))
test_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=Batch,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=Batch,
    shuffle=False
)

def lr_lambda(current_epoch):
    if current_epoch < warmup_epoch:
        # 0에서 1까지 직선으로 증가
        return float(current_epoch) / float(max(1, warmup_epoch))
    
    progress = float(current_epoch - warmup_epoch) / float(max(1, Epoch - warmup_epoch))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

model = CroCO().to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
start_epoch = 0

def train():
    print('Train Start')
    best_avg_test_loss = float('inf')

    for epoch in range(start_epoch, Epoch + 1):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0

        batch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)

            optimizer.zero_grad()

            logits, isMasked = model(image1, image2)

            loss = cross_loss(logits, image1, isMasked)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{Epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss_total : {loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                image1 = batch['image1'].to(device)
                image2 = batch['image2'].to(device)

                logits, isMasked = model(image1, image2)

                loss = cross_loss(logits, image1, isMasked)

                test_loss += loss.item()

        if epoch % 10 == 0:
            with torch.no_grad():
                sample = test_dataset[0]
                image1 = sample['image1'].unsqueeze(0).to(device)
                image2 = sample['image2'].unsqueeze(0).to(device)

                logits, isMasked = model(image1, image2)

                reconstruction_image = logit2image(logits[0], image1[0])

                masked_image = image2masked(image1[0], isMasked[0])

                image1 = to_imshow(image1[0])
                masked_image = to_imshow(masked_image)
                reconstruction_image = to_imshow(reconstruction_image)

                total_image = np.hstack((image1, masked_image, reconstruction_image))
                total_image = (total_image * 255).astype(np.uint8)
                total_img_pil = Image.fromarray(total_image)

                save_filename = f'result_{epoch}.png'
                total_img_pil.save(os.path.join(img_save_path, save_filename))
                
                print(f"==> Visualized result saved at {save_filename}")

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        epoch_end_time = time.time()

        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Test Loss : {avg_test_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % save_interval == 0:

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_test_loss': avg_test_loss,
            }

            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            torch.save(save_dict, save_path)
            
            print(f'Saved : {model_save_path}')
        
        if avg_test_loss < best_avg_test_loss:
            best_avg_test_loss = avg_test_loss

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 
                'avg_test_loss': avg_test_loss,
            }

            save_path = os.path.join(model_save_path, f'best_model_epoch.pth')
            torch.save(save_dict, save_path)

            print(f'New Best Model Saved! Loss : {best_avg_test_loss:.4f}')    

        scheduler.step()    

if __name__ == "__main__":
    train()