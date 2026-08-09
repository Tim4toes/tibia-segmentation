# This script generates Regions of Interest (ROIs) for grayscale bone segmentation using a pre-trained Attention U-Net. 
# OPTIMIZED FOR 240GB SYSTEM RAM & 24GB VRAM.

import os
import cv2
import numpy as np
import torch
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import your U-Net architecture from the master grayscale training script
from train_model import UNet 

# =============================================================================
# --- 1. DATASET & DATALOADER (RAM OPTIMIZED) ---
# WHAT IT DOES: Instead of processing a single image at a time, this establishes 
# an asynchronous pipeline. It caches the entire inference dataset into RAM and 
# queues up batches for the GPU.
# WHY IT IS NEEDED: Maximizes hardware utilization. 8 CPU workers handle the 
# resize/normalization math concurrently, allowing the 24GB VRAM GPU to process 
# 5 images simultaneously without waiting for disk latency.
# =============================================================================
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.ram_cache = []
        
        print(f"Caching {len(self.image_paths)} inference images into System RAM...")
        for img_path in tqdm(self.image_paths, desc="Loading to RAM"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            # Store original dimensions so we can seamlessly reshape predictions later
            self.ram_cache.append((img, img.shape[0], img.shape[1], str(img_path)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img, original_height, original_width, img_path = self.ram_cache[idx]
        augmented = self.transform(image=img)
        return augmented['image'], original_height, original_width, img_path

# =============================================================================
# --- 2. MODEL INITIALISATION ---
# =============================================================================
def generate_rois(model_weights_path, input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    if not os.path.exists(model_weights_path):
        print(f"Error: Could not find {model_weights_path}. Run train_model.py first.")
        return
        
    checkpoint = torch.load(model_weights_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

# =============================================================================
# --- 3. DIRECTORY AND TRANSFORM SETUP ---
# =============================================================================
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    infer_transform = A.Compose([
        A.Resize(960, 960, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.0330], std=[0.0726], max_pixel_value=255.0),
        ToTensorV2(),
    ])

# =============================================================================
# --- 4. FILE DISCOVERY ---
# =============================================================================
    print(f"Scanning for .bmp files in {input_dir}...")
    image_paths = list(input_dir.rglob("*.bmp"))
    
    if not image_paths:
        print(f"No .bmp files found. Please check your subfolders in {input_dir}.")
        return

    print(f"Generating ROIs for {len(image_paths)} images...")

# =============================================================================
# --- 5. THE BATCHED INFERENCE LOOP ---
# WHAT IT DOES: Processes 5 images through the AI at once. Squeezes the channel 
# dimension, extracts the binary arrays, rescales them back to their native height 
# and width, applies the median smoothing filter, and commits them back to disk.
# =============================================================================
    dataset = InferenceDataset(image_paths, infer_transform)
    # Using batch_size=5 for the 24GB VRAM ceiling, and num_workers=8 for the massive CPU RAM
    loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)
    
    for batch_tensors, batch_heights, batch_widths, batch_paths in tqdm(loader, desc="AI Inference"):
        batch_tensors = batch_tensors.to(device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                predictions = model(batch_tensors)
                # Squeeze the channel dimension but keep the batch dimension intact [Batch, 960, 960]
                prob_masks = torch.sigmoid(predictions).squeeze(1).cpu().numpy()
        
        # Iteratively post-process and save each individual image in the batch
        for i in range(len(batch_paths)):
            # Binarize
            binary_mask = (prob_masks[i] > 0.5).astype(np.uint8) * 255
            
            # Reshape back to native dimensions
            h, w = batch_heights[i].item(), batch_widths[i].item()
            final_roi_8bit = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Apply morphological smoothing
            final_roi_8bit = cv2.medianBlur(final_roi_8bit, 3)
            
            # Reconstruct the output directory tree
            img_path = Path(batch_paths[i])
            relative_path = img_path.relative_to(input_dir)
            out_path = output_dir / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save strictly as 1-bit monochrome
            final_roi_img = Image.fromarray(final_roi_8bit).convert('1')
            final_roi_img.save(str(out_path))
            
    print(f"Success! All {len(image_paths)} ROIs generated and mirrored seamlessly in {output_dir}")

# =============================================================================
# --- 6. TERMINAL PARSER ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROIs using trained Grayscale Attention U-Net")
    
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        choices=["tibia", "cortical", "trabecular"],
        help="Select the biological target to segment."
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="Optional: Override the default input directory with an external path."
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Optional: Override the default output directory with an external path."
    )
    
    args = parser.parse_args()
    
    if args.target == "tibia":
        model_weights_path = "checkpoints/tibia_unet.pth"
        default_input = "data/inference/input_tibia"
        default_output = "data/inference/output_tibia"
        
    elif args.target == "cortical":
        model_weights_path = "checkpoints/cortical_unet.pth"
        default_input = "data/inference/input_tibia_voi" 
        default_output = "data/inference/output_tibia_cortical" 
        
    elif args.target == "trabecular":
        model_weights_path = "checkpoints/trabecular_unet.pth"
        default_input = "data/inference/input_tibia_voi"       
        default_output = "data/inference/output_tibia_trabecular" 

    input_dir = args.input if args.input else default_input
    output_dir = args.output if args.output else default_output

    print(f"--- INITIALIZING INFERENCE PIPELINE ---")
    print(f"Target: {args.target.upper()}")
    print(f"Model: {model_weights_path}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    generate_rois(model_weights_path, input_dir, output_dir)