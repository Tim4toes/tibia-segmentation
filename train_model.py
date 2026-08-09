# This script trains an Attention U-Net model for grayscale bone segmentation.
# It includes a custom Dataset class that handles subfolder structures, applies textural augmentations, and saves the trained model for later inference.
# It includes GPU-accelerated overlap metrics (DSC, IoU, Sens, Prec) every epoch,
# and calculates CPU-intensive HD95 strictly when a new best model is saved.

import os
import cv2
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
from pathlib import Path
import random
from medpy.metric.binary import hd95

# =============================================================================
# --- 1. STRATIFIED SPLITTER ---
# WHAT IT DOES: Scans the master image directory, identifies individual datasets 
# based on their folder names (e.g., separating wildtype from knockout models), 
# and randomly divides them into training and validation groups.
# WHY IT IS NEEDED: Prevents "data leakage." If slices from the same physical bone 
# end up in both the training and validation sets, the AI will cheat by memorizing 
# the specific bone rather than learning general anatomy. This ensures the model 
# is always tested against entirely unseen morphology.
# =============================================================================
def get_stratified_split(images_base_dir, val_samples_per_group=1, forced_val_folders=None):
    image_dir = Path(images_base_dir)
    dataset_folders = [f for f in image_dir.iterdir() if f.is_dir()]
    
    train_folders = []
    val_folders = []

    if forced_val_folders is not None:
        print(f"Locking validation to previously saved folders: {forced_val_folders}")
        for folder in dataset_folders:
            if folder.name in forced_val_folders:
                val_folders.append(folder)
            else:
                train_folders.append(folder)
        return train_folders, val_folders

    groups = {}
    for folder in dataset_folders:
        group_name = folder.name.split('_')[-1].lower()
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(folder)
        
    for group, folders in groups.items():
        random.shuffle(folders) 
        if len(folders) <= val_samples_per_group:
            print(f"Warning: Group '{group}' only has {len(folders)} dataset(s). Assigning to training.")
            train_folders.extend(folders)
        else:
            val_folders.extend(folders[:val_samples_per_group])
            train_folders.extend(folders[val_samples_per_group:])
            
    print(f"New validation sets randomly selected: {[f.name for f in val_folders]}")
    return train_folders, val_folders

# =============================================================================
# --- 2. DATA LOADER ---
# WHAT IT DOES: Creates a dynamic index of file paths instead of loading all 
# images into memory at once. During training, it grabs an image, pairs it 
# with its ground-truth mask, applies augmentations, and converts the mask to a binary format.
# WHY IT IS NEEDED: MicroCT datasets contain thousands of high-resolution images. 
# Attempting to load them all simultaneously would instantly crash your 16GB of system RAM. 
# This class acts as a highly efficient conveyor belt, feeding data to the GPU batch by batch.
# =============================================================================
class BoneDataset(Dataset):
    def __init__(self, folder_list, mask_base_dir, transform=None):
        self.mask_base_dir = Path(mask_base_dir)
        self.transform = transform
        self.image_paths = []
        
        for folder in folder_list:
            self.image_paths.extend(list(folder.rglob("*.bmp")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
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

# =============================================================================
# --- 3. IMAGE AUGMENTATIONS ---
# WHAT IT DOES: Alters the images in real-time before they hit the network. It 
# dynamically warps shapes, flips orientations, injects static noise, shifts contrast, 
# and normalizes the pixel distribution.
# WHY IT IS NEEDED: Prevents overfitting. By constantly changing the visual presentation 
# of the bone, the AI cannot memorize specific pixels. The textural augmentations 
# specifically force the model to rely on actual tissue density differences (Hounsfield units) 
# rather than scanning artifacts.
# =============================================================================
train_transform = A.Compose([
    A.Resize(960, 960, interpolation=cv2.INTER_NEAREST),
    A.Rotate(limit=35, p=0.8, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5, interpolation=cv2.INTER_NEAREST),
    A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST),
    
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    
    # Dataset Z-Score Normalization
    A.Normalize(mean=[0.0330], std=[0.0726], max_pixel_value=255.0), 
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(960,960, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.0330], std=[0.0726], max_pixel_value=255.0), 
    ToTensorV2(),
])

# =============================================================================
# --- 4. ATTENTION U-NET ARCHITECTURE ---
# WHAT IT DOES: Builds the neural network blueprint. The encoder extracts features 
# as the image shrinks, the decoder reconstructs the image using those features, and 
# the Attention Gates filter the skip connections.
# WHY IT IS NEEDED: Standard U-Nets pass all visual data equally. Attention Gates 
# learn to highlight salient features (like dense cortical bone edges) while muting 
# irrelevant background noise (like air or soft tissue), resulting in much sharper boundaries.
# =============================================================================
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

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ag = nn.ModuleList() 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            self.ag.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))

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
            g = self.ups[idx](x) 
            skip_connection = skip_connections[idx//2]
            x_attended = self.ag[idx//2](g=g, x=skip_connection)
            concat_skip = torch.cat((x_attended, g), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# =============================================================================
# --- 5. HYBRID LOSS FUNCTION ---
# WHAT IT DOES: Combines Pixel-wise Binary Cross Entropy (BCE) with Global Dice Loss.
# WHY IT IS NEEDED: BCE evaluates each pixel independently, which can cause the AI 
# to become lazy if 90% of the scan is empty background space. Dice Loss calculates 
# the overall geometrical overlap of the bone prediction. Fusing them forces the model 
# to be highly accurate on complex boundaries (like trabecular struts).
# =============================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + 1e-6) / (probs_flat.sum() + targets_flat.sum() + 1e-6)
        dice_loss = 1 - dice
        
        return bce_loss + dice_loss

# =============================================================================
# --- 6. EVALUATION METRICS ---
# WHAT IT DOES: Compares the AI's output to your manual masks, generating scores 
# like DSC (overlap), Sensitivity (true positive rate), and HD95 (distance error).
# WHY IT IS NEEDED: These metrics tell you objectively if the model is getting better 
# or worse. HD95 is strictly sequestered behind a conditional toggle because it is 
# notoriously CPU-heavy and would stall training if calculated on every batch.
# =============================================================================
def calculate_metrics(pred_logits, true_masks,compute_hd95=False):
    preds = (torch.sigmoid(pred_logits) > 0.5).float()
    
    TP = (preds * true_masks).sum()
    FP = ((preds == 1) & (true_masks == 0)).sum()
    FN = ((preds == 0) & (true_masks == 1)).sum()
    
    dsc = (2. * TP) / (2. * TP + FP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
            
    batch_hd95 = np.nan
    
    if compute_hd95:
        hd95_list = []
        preds_np = preds.cpu().numpy()
        trues_np = true_masks.cpu().numpy()
        
        for i in range(preds_np.shape[0]):
            p = preds_np[i].squeeze()
            t = trues_np[i].squeeze()
            if p.max() > 0 and t.max() > 0:
                hd95_list.append(hd95(p, t))
                
        if len(hd95_list) > 0:
            batch_hd95 = np.mean(hd95_list)
            
    return dsc.item(), iou.item(), sensitivity.item(), precision.item(), batch_hd95

# =============================================================================
# --- 7. MAIN TRAINING LOOP ---
# WHAT IT DOES: The central command hub. It manages the 3-mode loading logic, 
# orchestrates the forward and backward passes (learning), triggers the validation 
# testing phase, manages the learning rate scheduler, and commits the best models to disk.
# WHY IT IS NEEDED: Automates the entire complex deep learning pipeline so you only 
# need to execute a single terminal command to handle training, testing, and logging.
# =============================================================================
def train_model(run_mode, epochs, model_path, images_base, masks_base, csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    start_epoch = 0
    best_loss = float('inf')
    forced_val_folders = None
    
    if run_mode in ["resume", "finetune"] and os.path.exists(model_path):
        print(f"\nLoading checkpoint for {run_mode.upper()} mode...")
        checkpoint = torch.load(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'val_folders' in checkpoint:
                forced_val_folders = checkpoint['val_folders']
            
            if run_mode == "resume":
                learning_rate = 1e-4
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler = torch.amp.GradScaler('cuda')
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint['best_loss']
                print(f"Resuming exactly from Epoch {start_epoch} with retained momentum.")
                
            elif run_mode == "finetune":
                learning_rate = 1e-5
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scaler = torch.amp.GradScaler('cuda')
                print("Fine-tuning: Learning rate dropped, optimizer momentum reset to zero.")
                
        else:
            model.load_state_dict(checkpoint)
            learning_rate = 1e-5 if run_mode == "finetune" else 1e-4
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scaler = torch.amp.GradScaler('cuda')
            print("Loaded legacy weights. Validation split will be random.")
    else:
        print("\nStarting BRAND NEW training run...")
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler('cuda')

    train_folders, val_folders = get_stratified_split(images_base, forced_val_folders=forced_val_folders)
    val_folder_names = [f.name for f in val_folders]
    
    train_dataset = BoneDataset(train_folders, masks_base, transform=train_transform)
    val_dataset = BoneDataset(val_folders, masks_base, transform=val_transform)

    # CRITICAL OPTIMIZATION: num_workers reduced to 2 for train, 1 for val to prevent 16GB RAM crash
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BCEDiceLoss() 
    scaler = torch.amp.GradScaler('cuda') 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
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
            
            with torch.amp.autocast('cuda'):
                predictions = model(data)
                loss = criterion(predictions, targets)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_metrics = {"dsc": 0, "iou": 0, "sens": 0, "prec": 0}
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.float().unsqueeze(1).to(device)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(data)
                    loss = criterion(predictions, targets)
                    
                val_loss += loss.item()
                
                dsc, iou, sens, prec, _ = calculate_metrics(predictions, targets, compute_hd95=False)
                val_metrics["dsc"] += dsc
                val_metrics["iou"] += iou
                val_metrics["sens"] += sens
                val_metrics["prec"] += prec
                
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
            avg_dsc = val_metrics["dsc"] / len(val_loader)
            avg_iou = val_metrics["iou"] / len(val_loader)
            avg_sens = val_metrics["sens"] / len(val_loader)
            avg_prec = val_metrics["prec"] / len(val_loader)
            print(f"Loss: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")
            print(f"Validation Metrics: DSC {avg_dsc:.4f} | IoU {avg_iou:.4f} | Sens {avg_sens:.4f} | Prec {avg_prec:.4f}")
            
            current_eval_loss = avg_val_loss
            loss_type = "Val Loss"
            scheduler.step(current_eval_loss)

        else:
            print(f"Avg Train Loss: {avg_train_loss:.4f} | (No validation data available)")
            current_eval_loss = avg_train_loss
            loss_type = "Train Loss"
        
        # --- BEST MODEL CHECKPOINT/SAVING ---
        if current_eval_loss < best_loss:
            best_loss = current_eval_loss

            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'val_folders': val_folder_names 
            }

            torch.save(checkpoint_dict, model_path)

            print(f"*** New best model found! Calculating HD95 across validation set... ***")
            hd95_scores = []

            with torch.no_grad():
                hd95_loop = tqdm(val_loader, desc="Calculating HD95", leave=False)
                
                for data, targets in hd95_loop:
                    data = data.to(device)
                    targets = targets.float().unsqueeze(1).to(device)
                    
                    with torch.amp.autocast('cuda'):
                        predictions = model(data)
                        
                    _, _, _, _, batch_hd95 = calculate_metrics(predictions, targets, compute_hd95=True)
                    if not np.isnan(batch_hd95):
                        hd95_scores.append(batch_hd95)
                        hd95_loop.set_postfix(batch_hd95=f"{batch_hd95:.2f} px")
                        
            avg_hd95 = np.mean(hd95_scores) if len(hd95_scores) > 0 else float('nan')
            print(f"*** Best Model Saved ({loss_type}: {best_loss:.4f}) | HD95: {avg_hd95:.2f} px ***")

            # --- APPEND METRICS TO CSV ---
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='') as csvfile:
                fieldnames = ['Epoch', 'Train_Loss', 'Val_Loss', 'DSC', 'IoU', 'Sensitivity', 'Precision', 'HD95']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerow({
                    'Epoch': epoch + 1,
                    'Train_Loss': f"{avg_train_loss:.4f}",
                    'Val_Loss': f"{best_loss:.4f}",
                    'DSC': f"{avg_dsc:.4f}",
                    'IoU': f"{avg_iou:.4f}",
                    'Sensitivity': f"{avg_sens:.4f}",
                    'Precision': f"{avg_prec:.4f}",
                    'HD95': f"{avg_hd95:.2f}"
                })

# =============================================================================
# --- 8. TERMINAL PARSER ---
# WHAT IT DOES: intercepts the commands you type in your command prompt and 
# dynamically assigns the correct variables before running the train_model() function.
# WHY IT IS NEEDED: Prevents you from having to manually hardcode file paths every 
# time you want to switch between training a macro tibia model and a trabecular model.
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention U-Net for Grayscale Bone Segmentation")
    
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        choices=["tibia", "cortical", "trabecular"],
        help="Select the biological target to segment."
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="new", 
        choices=["new", "resume", "finetune"],
        help="Select the training mode: new (scratch), resume (continue), or finetune (add data)."
    )
    
    args = parser.parse_args()
    
    if args.target == "tibia":
        model_path = "checkpoints/tibia_unet.pth"
        images_base = "data/macro_tibia/images"
        masks_base = "data/macro_tibia/masks"
        csv_path = "logs/metrics_tibia.csv"
        
    elif args.target == "cortical":
        model_path = "checkpoints/cortical_unet.pth"
        images_base = "data/tibia_voi/images"
        masks_base = "data/tibia_voi/masks_cortical"
        csv_path = "logs/metrics_cortical.csv"
        
    elif args.target == "trabecular":
        model_path = "checkpoints/trabecular_unet.pth"
        images_base = "data/tibia_voi/images"
        masks_base = "data/tibia_voi/masks_trabecular"
        csv_path = "logs/metrics_trabecular.csv"

    print(f"--- INITIALIZING PIPELINE ---")
    print(f"Target: {args.target.upper()}")
    print(f"Mode: {args.mode.upper()}")
    
    train_model(
        run_mode=args.mode, 
        epochs=50, 
        model_path=model_path, 
        images_base=images_base, 
        masks_base=masks_base, 
        csv_path=csv_path
    )

# How to run different training modes:
# Open terminal (CMD or PowerShell) and navigate to the project directory. 
# Then execute one of the following commands:

# 1 - Starting a brand-new tibia model:
# python train_model.py --target tibia --mode new

# 2 - Adding new datasets to your existing tibia model:
# python train_model.py --target tibia --mode finetune

# 3 - Resuming a crashed tibia training run:
# python train_model.py --target tibia --mode resume

# What the models do:
# train_model(run_mode="new", epochs=50): 
# Generates a completely new random split, starts at Epoch 0, sets LR to 1e-4.

# train_model(run_mode="resume", epochs=50): 
# Reads your last checkpoint, locks in the exact same validation datasets, 
# loads your optimizer momentum, and picks up exactly on the epoch where you cancelled it.

# train_model(run_mode="finetune", epochs=100): 
# Reads your last checkpoint, locks in the validation datasets, 
# drops the LR to 1e-5, resets the epoch counter to 0, 
# and begins delicate training (perfect for when you drop new datasets into your folders).