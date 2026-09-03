# This script generates Regions of Interest (ROIs) for grayscale bone segmentation.
# OPTIMIZED FOR: Multi-threaded batch inference (Dual Xeon / RTX A2000 12GB)

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

from train_model import UNet 

# =============================================================================
# --- CUSTOM INFERENCE DATASET ---
# WHAT IT DOES: Feeds the DataLoader by reading and transforming images on the CPU.
# =============================================================================
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        original_height, original_width = original_img.shape

        augmented = self.transform(image=original_img)
        img_tensor = augmented['image']
        
        return img_tensor, str(img_path), original_height, original_width

def generate_rois(model_weights_path, input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    if not os.path.exists(model_weights_path):
        print(f"Error: Could not find {model_weights_path}. Run train_model.py first.")
        return
        
    checkpoint = torch.load(model_weights_path, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    infer_transform = A.Compose([
        A.Resize(960, 960, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.0330], std=[0.0726], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    print(f"Scanning for .bmp files in {input_dir}...")
    image_paths = list(input_dir.rglob("*.bmp"))
    
    if not image_paths:
        print(f"No .bmp files found. Please check your subfolders in {input_dir}.")
        return

    print(f"Generating ROIs for {len(image_paths)} images using batch processing...")

    # =============================================================================
    # --- BATCHED INFERENCE LOADER ---
    # Optimized for 384GB RAM and 40-thread Xeon CPUs.
    # Batch size 4 maxes out A2000 VRAM efficiency for 960x960 tensors.
    # =============================================================================
    dataset = InferenceDataset(image_paths, infer_transform)
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=2
    )

    for img_tensors, paths, heights, widths in tqdm(loader, desc="Processing Batches"):
        img_tensors = img_tensors.to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                predictions = model(img_tensors)
                prob_masks = torch.sigmoid(predictions).cpu().numpy()

        # Iterate through the batch to resize and save locally
        for i in range(len(paths)):
            img_path = Path(paths[i])
            orig_h = heights[i].item()
            orig_w = widths[i].item()
            
            prob_mask = prob_masks[i].squeeze()
            binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
            
            final_roi_8bit = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            final_roi_8bit = cv2.medianBlur(final_roi_8bit, 3)

            relative_path = img_path.relative_to(input_dir)
            out_path = output_dir / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            final_roi_img = Image.fromarray(final_roi_8bit).convert('1')
            final_roi_img.save(str(out_path))
        
    print(f"Success! All {len(image_paths)} ROIs generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROIs using trained Grayscale Attention U-Net")
    parser.add_argument("--target", type=str, required=True, choices=["tibia", "cortical", "trabecular"])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
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
    
    generate_rois(model_weights_path, input_dir, output_dir)