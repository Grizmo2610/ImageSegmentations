# %%
from pycocotools.coco import COCO

import gc
import re
import os
import cv2
import time
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# %%
destination_dir = './content/data'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

kaggle_notebook = False

if kaggle_notebook:
    data_path = '/kaggle/input/coco-2017-dataset'
else:
    import kagglehub
    path = kagglehub.dataset_download('awsaf49/coco-2017-dataset')
    data_path = shutil.move(path, destination_dir)
    print(f'Data moved to: {data_path}')
    print('Data source import complete.')

# %%
FOLDER_PATH = os.path.join(data_path, 'coco2017/')
MODEL_FOLDER = 'models'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

IMAGE_SIZE = 224
NUM_WORKER = os.cpu_count()
learning_rate=1e-4
epochs = 5
best_val_loss = float("inf")
patience = 3
counter = 0

if torch.cuda.is_available():
    BATCH_SIZE = 8
    device = torch.device("cuda")
else:
    BATCH_SIZE = 128
    device = torch.device("cpu")

print(f'Device using: {device}')
print(f'CPU Count: {NUM_WORKER}')

# %%
class COCOSegmentation(Dataset):
    def __init__(self, root, annotation_file, transform=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load segmentation mask
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in annotations:
            mask = np.maximum(mask, self.coco.annToMask(ann) * ann["category_id"])

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.long()

    def __len__(self):
        return len(self.ids)

# Augmentations
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Load datasets
print(f'{"=" * 25}Loading Training dataset{"=" * 25}')
train_dataset = COCOSegmentation(os.path.join(FOLDER_PATH, "train2017"), 
                                 os.path.join(FOLDER_PATH, "annotations/instances_train2017.json"),
                                 transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)


print(f'{"=" * 25}Loading Validation dataset{"=" * 25}')
val_dataset = COCOSegmentation(os.path.join(FOLDER_PATH, "val2017"), 
                                 os.path.join(FOLDER_PATH, "annotations/instances_val2017.json"), 
                               transform=val_transform)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=91):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)

# %%
model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_unet = os.path.join(MODEL_FOLDER,'best_unet.pth')

# %%
paths = os.listdir(MODEL_FOLDER)
model_list = sorted([path.replace('10', '10') for path in paths if path.endswith('.pth')])
epoch_models = [(int(re.search(r'(\d+)', model).group()), model) for model in model_list if re.search(r'unet_epoch_(\d+)\.pth', model)]
if len(epoch_models) < 1:
    latest_model = 0
else:
    latest_model = max(epoch_models, key=lambda x: x[0])[1]
  
print(latest_model)

# %%
using_best = False
if using_best:
    model_path = best_unet
else:
    model_path = os.path.join(MODEL_FOLDER, latest_model)
print(f'Model using: {model_path}')

# %%
if os.path.exists(model_path):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    print(f"Model loaded successfully on {device}")
else:
    print(f"Error: Model path '{model_path}' does not exist.")

# %%
def calculate_iou(pred_mask, true_mask, num_classes=91):
    iou_scores = []
    for cls in range(1, num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        intersection = (pred_cls * true_cls).sum()
        union = (pred_cls + true_cls).sum()

        if union == 0:
            iou_scores.append(float("nan"))
        else:
            iou_scores.append((intersection / union).item())

    return np.nanmean(iou_scores)

def calculate_dice(pred_mask, true_mask, num_classes=91):
    dice_scores = []
    for cls in range(1, num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        intersection = (pred_cls * true_cls).sum()
        dice = (2.0 * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-6)

        dice_scores.append(dice.item())

    return np.nanmean(dice_scores)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    iou_scores, dice_scores, val_losses = [], [], []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, masks)
            val_losses.append(loss.item())

            iou = calculate_iou(preds, masks)
            dice = calculate_dice(preds, masks)

            iou_scores.append(iou)
            dice_scores.append(dice)

    avg_loss = np.mean(val_losses)
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)

    print(f"\nEvaluation - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
    return avg_loss, avg_iou, avg_dice  

# %%
if torch.cuda.is_available() and os.path.exists(model_path):
    avg_loss, avg_iou, avg_dice = evaluate_model(model, val_loader, criterion, device)
    print(f"Evaluation - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")  

# %%
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc="Training", leave=False)

    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loop = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(val_loader)

# %%
for epoch in range(epochs):
    print('=' * 25 + f'Epoch {epoch + 1}/ {epochs}' + '=' * 25)
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s - Release: {gc.collect()} objects")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, 'best_unet.pth'))
        print(f'New Best model Saved at {os.path.join(MODEL_FOLDER, "best_unet.pth")}')
    else:
        counter += 1
        print(f"Early Stopping Counter: {counter}/{patience}")
    
    saved_models = [f for f in os.listdir(MODEL_FOLDER) if re.match(r'unet_epoch_\d+\.pth', f)]
    max_epoch = 0
    if saved_models:
        max_epoch = max(int(re.search(r'unet_epoch_(\d+)\.pth', f).group(1)) for f in saved_models)
    
    next_epoch = max_epoch + 1
    torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, f"unet_epoch_{next_epoch}.pth"))
    print(f"Saved Model: unet_epoch_{next_epoch}.pth")
    
    if counter >= patience:
        print("Early stopping triggered! Training stopped.")
        break

# %%
if torch.cuda.is_available():
    model.eval()
    evaluate_model(model, val_loader, criterion, device)