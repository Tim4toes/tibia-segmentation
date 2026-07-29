# This script trains a U-Net model for bone segmentation using images and masks stored in subfolders.
# It includes a custom Dataset class that handles subfolder structures, applies augmentations, and saves the trained model for later inference.

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path
import random

# --- 1. STRATIFIED SPLITTER ---
# function scans your images directory and groups your datasets based on the word following 
# the final underscore (e.g., wildtype). It randomly selects a specified number of datasets 
# from each group (default is 1) to set aside for the validation "pop quiz," ensuring the 
# model is always tested on diverse anatomy.

def get_stratified_split(images_base_dir, val_samples_per_group=1):
    image_dir = Path(images_base_dir)
    dataset_folders = [f for f in image_dir.iterdir() if f.is_dir()]
    
    groups = {}
    for folder in dataset_folders:
        # Extracts the text after the last underscore
        group_name = folder.name.split('_')[-1].lower()
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(folder)
        
    train_folders = []
    val_folders = []
    
    for group, folders in groups.items():
        random.shuffle(folders) # Randomize dataset selection
        
        # Keep groups with only 1 dataset in the training pool
        if len(folders) <= val_samples_per_group:
            print(f"Warning: Group '{group}' only has {len(folders)} dataset(s). Assigning to training.")
            train_folders.extend(folders)
        else:
            val_folders.extend(folders[:val_samples_per_group])
            train_folders.extend(folders[val_samples_per_group:])
            
    print(f"Validation sets selected for testing: {[f.name for f in val_folders]}")
    return train_folders, val_folders

# --- 2. DATA LOADER ---
# class prevents your 16GB of system RAM from crashing. Instead of loading every .bmp 
# into memory at once, it creates a list of file paths. During training, it iteratively 
# grabs a few images from your hard drive, matches them to their exact ground-truth 
# mask, processes them, hands them to the GPU, and then clears them from memory.

class BoneDataset(Dataset):
    def __init__(self, folder_list, mask_base_dir, transform=None):
        self.mask_base_dir = Path(mask_base_dir)
        self.transform = transform
        self.image_paths = []
        
        # Collect all .bmp files from the assigned folders
        for folder in folder_list:
            self.image_paths.extend(list(folder.rglob("*.bmp")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Match the image to its exact mask in the masks directory
        folder_name = img_path.parent.name
        file_name = img_path.name
        mask_path = self.mask_base_dir / folder_name / file_name
        
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
             raise FileNotFoundError(f"Missing mask for {img_path}")
                
        # Binarize mask for Tissue Volume envelope (0 = background, 1 = bone ROI)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            
        return image, mask

# --- 3. IMAGE AUGMENTATIONS ---
# instructions tell Albumentations how to modify the images on the fly. train_transform 
# shrinks the image to fit your GPU, randomly rotates it, and flips it to artificially 
# multiply your training data. val_transform only shrinks and normalizes the image, 
# ensuring the validation test is performed on an unaltered, "clean" scan.


# Resize to 512x512 to save VRAM, plus augmentations
train_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Rotate(limit=35, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), 
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), 
    ToTensorV2(),
])

# --- 4. U-NET ARCHITECTURE ---
# defines the structural "brain" of the neural network. The encoder (downs) shrinks 
# the image to locate bone context, the bottleneck processes the densest features, and 
# the decoder (ups) expands the image back out to draw precise pixel boundaries using 
# skip connections.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- 5. TRAINING LOOP ---
# central command block that orchestrates everything. It triggers the data splitter, 
# initializes the GPU (utilizing mixed precision autocast to maximize your RTX 3090's 
# memory), runs the training pass, runs the validation test pass, and saves the .pth 
# weights only if the validation score has improved.

def train_model(resume_training=False, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model_path = "tibia_unet.pth"
    
    # Check if we are fine-tuning an existing model or starting fresh
    if resume_training and os.path.exists(model_path):
        print(f"Loading existing model weights...")
        model.load_state_dict(torch.load(model_path))
        learning_rate = 1e-5
    else:
        print("Starting training from a blank slate...")
        learning_rate = 1e-4

    # Execute the stratified split
    images_base = "data/all_datasets/images"
    masks_base = "data/all_datasets/masks"
    train_folders, val_folders = get_stratified_split(images_base)
    
    # Initialize the data loaders
    train_dataset = BoneDataset(train_folders, masks_base, transform=train_transform)
    val_dataset = BoneDataset(val_folders, masks_base, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Set up math optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() 
    
    best_loss = float('inf') 
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc="Training")
        
        for data, targets in loop:
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = criterion(predictions, targets)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE (The Pop Quiz) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.float().unsqueeze(1).to(device)
                
                with torch.cuda.amp.autocast():
                    predictions = model(data)
                    loss = criterion(predictions, targets)
                    
                val_loss += loss.item()
                
# Check if validation data exists to prevent ZeroDivisionError
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
            print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
            current_eval_loss = avg_val_loss
            loss_type = "Val Loss"
        else:
            print(f"Avg Train Loss: {avg_train_loss:.4f} | (No validation data available)")
            current_eval_loss = avg_train_loss
            loss_type = "Train Loss"
        
        # --- BEST MODEL SAVING ---
        if current_eval_loss < best_loss:
            best_loss = current_eval_loss
            torch.save(model.state_dict(), model_path)
            print(f"*** New best model saved! ({loss_type}: {best_loss:.4f}) ***")

# Entry point to execute the script
if __name__ == "__main__":
    train_model(resume_training=False, epochs=50)