# This script trains an Attention U-Net model for black-and-white bone segmentation.
# It includes a custom Dataset class that handles subfolder structures, applies augmentations, and saves the trained model for later inference.
# It includes GPU-accelerated overlap metrics (DSC, IoU, Sens, Prec) every epoch,
# and calculates CPU-intensive HD95 strictly when a new best model is saved (Strategy A).

# See instructions at end of script and in README.md for how to run the three training modes: 
# new, resume, and finetune.

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

# --- 1. STRATIFIED SPLITTER (Updated for Checkpoint Locking) ---
# function scans your images directory and groups your datasets based on the word following 
# the final underscore (e.g., wildtype). It randomly selects a specified number of datasets 
# from each group (default is 1) to set aside for the validation "pop quiz," ensuring the 
# model is always tested on diverse anatomy.

def get_stratified_split(images_base_dir, val_samples_per_group=1, forced_val_folders=None):
    image_dir = Path(images_base_dir)
    dataset_folders = [f for f in image_dir.iterdir() if f.is_dir()]
    
    train_folders = []
    val_folders = []

    # Mode A: If resuming or finetuning, lock validation to the previously saved folders
    if forced_val_folders is not None:
        print(f"Locking validation to previously saved folders: {forced_val_folders}")
        for folder in dataset_folders:
            if folder.name in forced_val_folders:
                val_folders.append(folder)
            else:
                train_folders.append(folder)
        return train_folders, val_folders

# Mode B: If starting a new run, do a brand new, truly random split
    groups = {}
    for folder in dataset_folders:
        # Extracts the text after the last underscore
        group_name = folder.name.split('_')[-1].lower()
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(folder)
        
    for group, folders in groups.items():
        random.shuffle(folders) # Unseeded, true random shuffle of validation dataset selection
        # Keep groups with only 1 dataset in the training pool
        if len(folders) <= val_samples_per_group:
            print(f"Warning: Group '{group}' only has {len(folders)} dataset(s). Assigning to training.")
            train_folders.extend(folders)
        else:
            val_folders.extend(folders[:val_samples_per_group])
            train_folders.extend(folders[val_samples_per_group:])
            
    print(f"New validation sets randomly selected: {[f.name for f in val_folders]}")
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

# Resize to 1024x1024 to save VRAM, plus augmentations
train_transform = A.Compose([
    # Force nearest-neighbor interpolation to prevent gray edge artifacts
    A.Resize(960, 960, interpolation=cv2.INTER_NEAREST),
    # Original spatial augmentations using nearest-neighbor
    A.Rotate(limit=35, p=0.8, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    # NEW: Shape-warping to prevent the AI from memorizing exact binary shapes
    A.ElasticTransform(alpha=1, sigma=50, p=0.5, interpolation=cv2.INTER_NEAREST),
    A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST),
    # Standard Normalization and Tensor conversion
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), 
    ToTensorV2(),
])

val_transform = A.Compose([
    # Ensure validation scans are also resized without introducing gray artifacts
    A.Resize(960,960, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), 
    ToTensorV2(),
])

# --- 4. ATTENTION U-NET ARCHITECTURE ---
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

class AttentionGate(nn.Module):
    # Acts as a filter. It uses the gating signal (g) from the decoder to 
    # highlight the important features in the spatial map (x) from the encoder, 
    # suppressing irrelevant background structures.
    
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
        # Combine the signals and apply activation
        psi = self.relu(g1 + x1)
        # Generate the attention coefficients (values between 0 and 1)
        psi = self.psi(psi)
        # Multiply the spatial map by the coefficients to mute irrelevant areas
        return x * psi

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ag = nn.ModuleList() # Attention Gates
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling)
        for feature in reversed(features):
            # Upsample the image
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # Convolutions after concatenation
            self.ups.append(DoubleConv(feature*2, feature))
            # Initialize the Attention Gate for this specific tier
            self.ag.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- ENCODER PATH ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]


        # --- DECODER PATH WITH ATTENTION ---
        for idx in range(0, len(self.ups), 2):
            # 1. Upsample the current feature map (this is our gating signal)
            g = self.ups[idx](x) 
            # 2. Grab the corresponding skip connection from the encoder
            skip_connection = skip_connections[idx//2]
            # 3. Pass both through the Attention Gate
            x_attended = self.ag[idx//2](g=g, x=skip_connection)
            # 4. Concatenate the filtered spatial map with the gating signal
            concat_skip = torch.cat((x_attended, g), dim=1)
            # 5. Process through the DoubleConv block
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- 4.5 EVALUATION METRICS ---
def calculate_metrics(pred_logits, true_masks,compute_hd95=False):
    # Only calculate the fast GPU overlap metrics here
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

# --- 5. TRAINING LOOP (3-Mode System) ---
# central command block that orchestrates everything. It triggers the data splitter, 
# initializes the GPU (utilizing mixed precision autocast to maximize your RTX 3090's 
# memory), runs the training pass, runs the validation test pass, and saves the .pth 
# weights only if the validation score has improved.

def train_model(run_mode, epochs, model_path, images_base, masks_base, csv_path):
    # run_mode options: "new", "resume", or "finetune"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    start_epoch = 0
    best_loss = float('inf')
    forced_val_folders = None
    
    # --- CHECKPOINT LOADER ---
    if run_mode in ["resume", "finetune"] and os.path.exists(model_path):
        print(f"\nLoading checkpoint for {run_mode.upper()} mode...")
        checkpoint = torch.load(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Extract the saved validation split so we can reuse it
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
            # Backwards compatibility: Loads the old weights-only .pth file
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

    # Pass the forced_val_folders to the splitter (will be None if "new")
    train_folders, val_folders = get_stratified_split(images_base, forced_val_folders=forced_val_folders)

    # Save the folder names as strings so we can pack them into the checkpoint later
    val_folder_names = [f.name for f in val_folders]
    
    # Initialize the data loaders
    train_dataset = BoneDataset(train_folders, masks_base, transform=train_transform)
    val_dataset = BoneDataset(val_folders, masks_base, transform=val_transform)

    # can increase or decrease batch size to increase or decrease strain on GPU
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=5, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)
    
    # Set up math optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') 
    
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
        
        # --- VALIDATION PHASE (The Pop Quiz) ---
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
                
                # Fast GPU metric evaluation
                dsc, iou, sens, prec, _ = calculate_metrics(predictions, targets, compute_hd95=False)
                val_metrics["dsc"] += dsc
                val_metrics["iou"] += iou
                val_metrics["sens"] += sens
                val_metrics["prec"] += prec
                
            # Check if validation data exists to prevent ZeroDivisionError
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
        else:
            print(f"Avg Train Loss: {avg_train_loss:.4f} | (No validation data available)")
            current_eval_loss = avg_train_loss
            loss_type = "Train Loss"
        
        # --- BEST MODEL CHECKPOINT/SAVING AND HD95 COMPUTATION ---
        if current_eval_loss < best_loss:
            best_loss = current_eval_loss

            # Save the full state dictionary, including the validation folder names
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'val_folders': val_folder_names # <--- The validation split is permanently saved here
            }

            torch.save(checkpoint_dict, model_path)

            # Compute HD95 exclusively when a new best checkpoint is achieved
            print(f"*** New best model found! Calculating HD95 across validation set... ***")
            hd95_scores = []

            # Run a second, quick forward pass to generate predictions one batch at a time
            with torch.no_grad():
                # Wrap the val_loader in tqdm for a live progress bar
                hd95_loop = tqdm(val_loader, desc="Calculating HD95", leave=False)
                
                for data, targets in hd95_loop:
                    data = data.to(device)
                    targets = targets.float().unsqueeze(1).to(device)
                    
                    with torch.amp.autocast('cuda'):
                        predictions = model(data)
                        
                    # Calculate HD95 immediately and let Python garbage collect the arrays
                    _, _, _, _, batch_hd95 = calculate_metrics(predictions, targets, compute_hd95=True)
                    if not np.isnan(batch_hd95):
                        hd95_scores.append(batch_hd95)
                        # Update the progress bar text to show the current batch score
                        hd95_loop.set_postfix(batch_hd95=f"{batch_hd95:.2f} px")
                        
            avg_hd95 = np.mean(hd95_scores) if len(hd95_scores) > 0 else float('nan')
            print(f"*** Best Model Saved ({loss_type}: {best_loss:.4f}) | HD95: {avg_hd95:.2f} px ***")

            # --- NEW: APPEND METRICS TO CSV ---
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='') as csvfile:
                fieldnames = ['Epoch', 'Train_Loss', 'Val_Loss', 'DSC', 'IoU', 'Sensitivity', 'Precision', 'HD95']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write the header only if the file is being created for the first time
                if not file_exists:
                    writer.writeheader()
                    
                # Append the new row of data for this best model
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

# Entry point to execute the script
if __name__ == "__main__":
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Train Attention U-Net for Bone Segmentation")
    
    # 2. Add an argument for the anatomical target
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        choices=["tibia", "cortical", "trabecular"],
        help="Select the biological target to segment."
    )
    
    # 3. Add an argument for the training mode (defaulting to 'new')
    parser.add_argument(
        "--mode", 
        type=str, 
        default="new", 
        choices=["new", "resume", "finetune"],
        help="Select the training mode: new (scratch), resume (continue), or finetune (add data)."
    )
    
    # 4. Parse the commands entered in the terminal
    args = parser.parse_args()
    
    # 5. Route the paths based on the chosen target
    if args.target == "tibia":
        model_path = "checkpoints/tibia_unet_bw.pth"
        images_base = "data/macro_tibia/images_bw"
        masks_base = "data/macro_tibia/masks_tibia"
        csv_path = "logs/metrics_tibia.csv"
        
    elif args.target == "cortical":
        model_path = "checkpoints/cortical_unet_bw.pth"
        images_base = "data/tibia_voi/images_bw"
        masks_base = "data/tibia_voi/masks_cortical_bw"
        csv_path = "logs/metrics_cortical.csv"
        
    elif args.target == "trabecular":
        model_path = "checkpoints/trabecular_unet_bw.pth"
        images_base = "data/tibia_voi/images_bw"
        masks_base = "data/tibia_voi/masks_trabecular_bw"
        csv_path = "logs/metrics_trabecular.csv"

    print(f"--- INITIALIZING PIPELINE ---")
    print(f"Target: {args.target.upper()}")
    print(f"Mode: {args.mode.upper()}")
    
    # 6. Pass these dynamic variables directly into your training function
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
# python train_model_bw.py --target tibia --mode new

# 2 - Adding new datasets to your existing tibia model:
# python train_model_bw.py --target tibia --mode finetune

# Resuming a crashed tibia training run:
# python train_model_bw.py --target tibia --mode resume

# If you ever forget what commands are available, you can simply type:
# python train_model_bw.py --help

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