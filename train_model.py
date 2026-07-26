import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- 1. DATA LOADER ---
class BoneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        # Load one by one to save system RAM
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask (0 and 1) for the Tissue Volume envelope
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            
        return image, mask

# Resize to 512x512 to save VRAM, plus augmentations
train_transform = A.Compose([
    A.Resize(512, 512),
    A.Rotate(limit=35, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), 
    ToTensorV2(),
])

# --- 2. U-NET MODEL ---
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

# --- 3. TRAINING LOOP ---
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    dataset = BoneDataset(image_dir="data/train/images", mask_dir="data/train/masks", transform=train_transform)
    # Batch size of 4 is safe for 512x512 images on a 24GB RTX 3090
    loader = DataLoader(dataset, batch_size=4, shuffle=True) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() # Mixed precision to save VRAM
    
    epochs = 20
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        loop = tqdm(loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = criterion(predictions, targets)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())
            
    torch.save(model.state_dict(), "tibia_unet.pth")
    print("Model saved to tibia_unet.pth")

if __name__ == "__main__":
    train_model()