# This script generates Regions of Interest (ROIs) for grayscale bone segmentation using a pre-trained Attention U-Net. 
# It processes images stored in subfolders, applies textural normalizations, slightly smooths the boundaries, 
# and saves the resulting 1-bit ROIs while maintaining the original directory structure.
# OPTIMISED FOR PC3: 32GB System RAM, 24GB VRAM (RTX 4090) & i7-14700K CPU.

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
# --- 1. DATASET & DATALOADER (32GB RAM OPTIMISED) ---
# WHAT IT DOES: Iteratively reads images from the disk instead of RAM caching.
# WHY IT IS NEEDED: Caching a 30GB dataset into 32GB of RAM will cause an OS crash. 
# This class acts as a highly efficient conveyor belt, pulling data from the drive 
# just in time for the CPU workers to augment it.
# =============================================================================
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Read directly from disk to protect the 32GB RAM ceiling
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Store original dimensions for seamlessly reshaping predictions later
        original_height, original_width = img.shape

        augmented = self.transform(image=img)
        img_tensor = augmented['image']
        
        return img_tensor, original_height, original_width, str(img_path)

# =============================================================================
# --- 2. MODEL INITIALISATION ---
# WHAT IT DOES: Initializes the PyTorch environment, builds the untrained Attention 
# U-Net architecture in GPU memory, and dynamically extracts and injects the learned 
# weights (from your .pth dictionary checkpoint) into the network.
# WHY IT IS NEEDED: Without this step, the network is an empty shell with randomized 
# weights that knows nothing about bone anatomy. Calling `model.eval()` is a critical 
# mathematical lock; it disables layers like Dropout and Batch Normalization, ensuring 
# the model strictly predicts on the new data rather than attempting to learn from it.
# =============================================================================
def generate_rois(model_weights_path, input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    if not os.path.exists(model_weights_path):
        print(f"Error: Could not find {model_weights_path}. Run train_model.py first.")
        return
        
    # Extract weights securely from the dictionary checkpoint format
    checkpoint = torch.load(model_weights_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Backwards compatibility in case an older weights-only file is loaded
        model.load_state_dict(checkpoint)

    model.eval()

# =============================================================================
# --- 3. DIRECTORY AND TRANSFORM SETUP ---
# WHAT IT DOES: Establishes the I/O paths and defines the mathematical transformations 
# applied to the raw inference images. It resizes the scan to 960x960 and normalizes 
# the pixel values using your dataset's global mean (0.0330) and standard deviation (0.0726).
# WHY IT IS NEEDED: A neural network expects data in the exact same mathematical state 
# it was trained on. If you feed it raw pixel values (0-255) instead of normalized 
# Z-score tensors, the Attention Gates will mathematically collapse and fail to recognize 
# the tissue density boundaries. The 960x960 resize ensures it safely clears your 24GB VRAM ceiling.
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
# WHAT IT DOES: Recursively crawls through all nested subfolders inside your defined 
# input directory to build a complete index of every single .bmp file.
# WHY IT IS NEEDED: Automates batch processing for complex experiments. Instead of 
# pointing the script to individual files, you can drop multiple cropped datasets 
# (e.g., dataset1_sost, dataset2_sost) into the master input folder, and the script 
# will seamlessly discover and process the entire queue.
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
    
    # PC3 OPTIMIZATION: batch_size=5 maxes the 24GB VRAM. 
    # num_workers=8 and prefetch_factor=3 leverages the i7-14700K multi-threading.
    loader = DataLoader(
        dataset, 
        batch_size=5, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True, 
        prefetch_factor=3
    )
    
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
# WHAT IT DOES: Intercepts console commands and dynamically assigns the correct 
# input, output, and model paths based on the biological target you select.
# WHY IT IS NEEDED: Drastically streamlines your workflow. It allows you to seamlessly 
# switch between processing macro whole-bone tibias or cropped trabecular VOIs directly 
# from the terminal prompt, without ever having to manually open and edit the Python script.
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
    
    # --- NEW OPTIONAL ARGUMENTS FOR EXTERNAL ROUTING ---
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
    
    # 1. Establish the default baseline paths based on the chosen target
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

    # 2. Override defaults if external paths were provided in the terminal
    input_dir = args.input if args.input else default_input
    output_dir = args.output if args.output else default_output

    print(f"--- INITIALIZING INFERENCE PIPELINE ---")
    print(f"Target: {args.target.upper()}")
    print(f"Model: {model_weights_path}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    generate_rois(model_weights_path, input_dir, output_dir)