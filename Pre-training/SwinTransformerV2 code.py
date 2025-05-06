import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.swin_transformer_v2 import SwinTransformerV2 
import sys
sys.path.append('Swin-Transformer')

IMG_SIZE = 256
BATCH_SIZE = 20
EPOCHS = 100
TRAIN_IMAGE_COUNT = 100000
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-4
PATIENCE = 3
NUM_CLASSES = 1  


import kagglehub  
path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
root_dir = os.path.join(path, "real_vs_fake", "real-vs-fake")
train_csv = os.path.join(path, "train.csv")
valid_csv = os.path.join(path, "valid.csv")
test_csv  = os.path.join(path, "test.csv")

df_train = pd.read_csv(train_csv)
df_valid = pd.read_csv(valid_csv)
df_test  = pd.read_csv(test_csv)

if len(df_train) > TRAIN_IMAGE_COUNT:
    df_train = df_train.sample(n=TRAIN_IMAGE_COUNT, random_state=42)

# Custom dataset class with flipped images

class FaceDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, "path"])
        image = Image.open(img_path).convert("RGB")
        label = float(self.df.loc[idx, "label"]) 

        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = FaceDataset(df_train, root_dir, transform=train_transform)
valid_dataset = FaceDataset(df_valid, root_dir, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = SwinTransformerV2(
    img_size=IMG_SIZE,
    drop_path_rate=DROPOUT_RATE,
    num_classes=NUM_CLASSES,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=8
)
device = torch.device('cuda:0')
torch.cuda.set_device(device)
model = model.to(device)

pretrained_path = "./swinv2_base_patch4_window8_256.pth" 
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights loaded.")

for param in model.parameters():
    param.requires_grad = False

# Adding additional layers 

in_features = model.head.in_features 
model.head = nn.Sequential(
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(in_features, NUM_CLASSES),
)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

criterion = nn.BCEWithLogitsLoss()  

# Adding history

best_val_loss = np.inf
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
checkpoint_path = "swinv2_model.pth"

def calc_accuracy(outputs, labels):
    outputs = outputs.to(device)
    labels = labels.to(device)
    preds = (outputs > 0.5).float()
    return (preds == labels.unsqueeze(1)).float().mean().item()

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    train_accuracies = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        accuracy = calc_accuracy(outputs, labels)
        train_accuracies.append(accuracy)
        if (i + 1) % 3125 == 0:  
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Avg train loss: {np.mean(train_losses):.4f}, Avg train accuracy: {np.mean(train_accuracies):.4f}")
    
    avg_train_loss = np.mean(train_losses)
    avg_train_acc = np.mean(train_accuracies)

    model.eval()
    val_losses = []
    val_accuracies = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_losses.append(loss.item())
            val_accuracies.append(calc_accuracy(outputs, labels))
            
    avg_val_loss = np.mean(val_losses)
    avg_val_acc = np.mean(val_accuracies)
    
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    # Adding  custom early-stopping mechanism
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Checkpoint saved!")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

history_dir = "./"
if not os.path.exists(history_dir):
    os.makedirs(history_dir)
with open(os.path.join(history_dir, "swinv2_history.json"), "w") as f:
    json.dump(history, f)
print(history)

torch.save(model.state_dict(), "./swinv2_base_patch4_window8_256.pth")

