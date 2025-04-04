import os
import re
import gc
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
from Models import UNet

# ========================== CONFIG ==========================
DATA_PATH = 'path/to/data'
FOLDER_PATH = os.path.join(DATA_PATH, 'coco2017/')
MODEL_FOLDER = 'models'
IMAGE_SIZE = 224
EPOCHS = 5
LEARNING_RATE = 1e-4
PATIENCE = 3

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

NUM_WORKERS = os.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8 if torch.cuda.is_available() else 128
MODEL_PATH = os.path.join(MODEL_FOLDER, 'best_unet.pth')

print(f'Device using: {DEVICE}')
print(f'CPU Count: {NUM_WORKERS}')

# ======================= DATASET PLACEHOLDER =======================
class CustemDataset(Dataset):
    pass

# ======================= TRANSFORM =======================
def get_transform(size: int, is_train: bool = False):
    base_transforms = [A.Resize(size, size)]
    if is_train:
        base_transforms += [
            A.RandomResizedCrop(size, size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ]
    base_transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(base_transforms)

# ======================= LOAD DATA =======================
def load_dataset(image_folder, label_path, transform):
    dataset = CustemDataset(image_folder, label_path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print("=" * 25 + " Loading Training dataset " + "=" * 25)
train_loader = load_dataset(
    os.path.join(FOLDER_PATH, "train2017"),
    os.path.join(FOLDER_PATH, "annotations/instances_train2017.json"),
    get_transform(IMAGE_SIZE, is_train=True)
)

print("=" * 25 + " Loading Validation dataset " + "=" * 25)
val_loader = load_dataset(
    os.path.join(FOLDER_PATH, "val2017"),
    os.path.join(FOLDER_PATH, "annotations/instances_val2017.json"),
    get_transform(IMAGE_SIZE, is_train=False)
)

# ======================= MODEL =======================
model = UNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"Model path '{MODEL_PATH}' does not exist.")

# ======================= TRAINING =======================
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# ======================= TRAIN LOOP =======================
best_val_loss = float("inf")
counter = 0

for epoch in range(EPOCHS):
    print("=" * 25 + f" Epoch {epoch+1}/{EPOCHS} " + "=" * 25)
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = validate(model, val_loader, criterion, DEVICE)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} - Time={elapsed:.2f}s - GC={gc.collect()}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"New Best Model Saved at {MODEL_PATH}")
    else:
        counter += 1
        print(f"Early Stopping Counter: {counter}/{PATIENCE}")

    # Save every epoch
    epoch_model_path = os.path.join(MODEL_FOLDER, f"unet_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Saved Model: {epoch_model_path}")

    if counter >= PATIENCE:
        print("Early stopping triggered! Training stopped.")
        break
    
# ======================= SELECT MODEL =======================
def load_model(model, path, device):
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        print(f"Loaded model from {path}")
        return model
    else:
        print(f"Model file not found: {path}")
        return None

print("\n" + "=" * 25 + " Model Selection " + "=" * 25)
use_best = input("Use best model? (y/n): ").strip().lower()

if use_best == 'y':
    final_model_path = MODEL_PATH
else:
    available_epochs = sorted([
        f for f in os.listdir(MODEL_FOLDER) if re.match(r'unet_epoch_(\d+)\.pth', f)
    ])
    if not available_epochs:
        print("No epoch models found.")
        final_model_path = MODEL_PATH
    else:
        print("Available epoch models:")
        for idx, f in enumerate(available_epochs):
            print(f"[{idx}] {f}")
        selected = input("Select model index: ").strip()
        try:
            selected_idx = int(selected)
            final_model_path = os.path.join(MODEL_FOLDER, available_epochs[selected_idx])
        except:
            print("Invalid input. Using best model.")
            final_model_path = MODEL_PATH

model = load_model(model, final_model_path, DEVICE)