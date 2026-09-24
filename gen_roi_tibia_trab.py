# NOTE: this script is a modified version of gen_roi_tibia.py. Modifications will be transferred to the original model 
# if the modifications are successful. 
# This script generates Regions of Interest (ROIs) for grayscale bone segmentation using a pre-trained Attention U-Net. 
# It processes 2D slices, stacks them into a 3D volume, applies Connected Component 
# Analysis to delete the fibula and ankle, and saves the cleaned 2D 1-bit ROIs.
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
from collections import defaultdict
import scipy.ndimage as ndimage
from skimage import measure
from torch.utils.data import Dataset, DataLoader

# Import your U-Net architecture from the master grayscale training script
from train_tibia import UNet 

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
    checkpoint = torch.load(model_weights_path, weights_only=False)
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
        # 1. Scale longest side to 960 and pad the missing space with black pixels
        A.LongestMaxSize(max_size=960, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=960, min_width=960, border_mode=cv2.BORDER_CONSTANT, value=0),
        
        # 2. Apply global grayscale Z-score normalization and convert to PyTorch tensor
        A.Normalize(mean=[0.0330], std=[0.0726], max_pixel_value=255.0),
        ToTensorV2(),
    ])

# =============================================================================
# --- 4. FILE DISCOVERY AND VOLUME GROUPING---
# WHAT IT DOES: Recursively crawls through all nested subfolders inside your defined 
# input directory to build a complete index of every single .bmp file.
# WHY IT IS NEEDED: Automates batch processing for complex experiments. Instead of 
# pointing the script to individual files, you can drop multiple cropped datasets 
# (e.g., dataset1_sost, dataset2_sost) into the master input folder, and the script 
# will seamlessly discover and process the entire queue.
# =============================================================================
    print(f"Scanning for .bmp files in {input_dir}...")
    all_image_paths = list(input_dir.rglob("*.bmp"))
    
    if not all_image_paths:
        print(f"No .bmp files found. Please check your subfolders in {input_dir}.")
        return

    # Group slices by their parent folder so they can be processed as unified 3D volumes
    volume_groups = defaultdict(list)
    for path in all_image_paths:
        volume_groups[path.parent].append(path)

    print(f"Found {len(volume_groups)} distinct 3D scan volumes to process.")

# =============================================================================
# --- 5. 3D INFERENCE AND CLEANUP LOOP ---
# WHAT IT DOES: Processes 5 images through the AI at once. Squeezes the channel 
# dimension, extracts the binary arrays, rescales them back to their native height 
# and width, applies the median smoothing filter, and commits them back to disk.
# =============================================================================
    cleaned_volumes_list = []
    
    for folder, paths in volume_groups.items():
        # Sort paths alphanumerically to guarantee correct Z-axis physical order
        paths = sorted(paths)
        print(f"\nProcessing Volume: {folder.name} ({len(paths)} slices)")
        
        volume_predictions = []
        original_shapes = []
        out_paths = []

        # Instantiate the PC3-Optimized DataLoader for this specific folder
        # PC3 OPTIMIZATION: batch_size=5 maxes the 24GB VRAM. 
        # num_workers=8 and prefetch_factor=3 leverages the i7-14700K multi-threading.
        dataset = InferenceDataset(paths, infer_transform)
        loader = DataLoader(
            dataset, 
            batch_size=5, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True, 
            prefetch_factor=3
        )

        # Step 5A: Batched 2D AI Inference
        for batch_tensors, batch_heights, batch_widths, batch_paths in tqdm(loader, desc="AI Prediction", leave=False):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    predictions = model(batch_tensors)
                    # Squeeze the channel dimension but keep the batch dimension intact [Batch, 960, 960]
                    prob_masks = torch.sigmoid(predictions).squeeze(1).cpu().numpy()
            
            # Iteratively post-process and save each individual image in the batch
            for i in range(len(batch_paths)):
                # Binarize directly into a fast 8-bit integer for RAM protection
                binary_mask = (prob_masks[i] > 0.5).astype(np.uint8)
                volume_predictions.append(binary_mask)
                
                # Store structural variables for 3D unpacking later
                original_shapes.append((batch_widths[i].item(), batch_heights[i].item()))
                
                img_path = Path(batch_paths[i])
                relative_path = img_path.relative_to(input_dir)
                out_path = output_dir / relative_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_paths.append(out_path)
    
        # Step 5B: 3D Stack and Connected Component Analysis
        # Wrapped in a 4-step manual progress bar
        with tqdm(total=4, desc="3D Morphological Cleanup", leave=False) as cleanup_pbar:
            
            # 1. Stack 2D lists into a contiguous 3D boolean array (Saves RAM)
            cleanup_pbar.set_postfix(step="Stacking Arrays")
            volume_3d = np.stack(volume_predictions).astype(bool)
            cleanup_pbar.update(1)
            
            # 2. Sever weak 1-voxel bridges between the tibia and fibula
            cleanup_pbar.set_postfix(step="Opening Filter")
            struct_element = np.ones((3, 3, 3), dtype=bool)
            opened_volume = ndimage.binary_opening(volume_3d, structure=struct_element)
            cleanup_pbar.update(1)
            
            # 3. Label all distinct 3D islands and isolate the "Core Tibia"
            cleanup_pbar.set_postfix(step="Isolating Volume")
            labels = measure.label(opened_volume, connectivity=1)
            
            if labels.max() > 0:
                largest_cc_id = np.argmax(np.bincount(labels.flat)[1:]) + 1
                core_tibia = (labels == largest_cc_id)
            else:
                core_tibia = opened_volume
            cleanup_pbar.update(1)

            # 4. RESTORE PRISTINE BOUNDARIES: Masked Dilation
            cleanup_pbar.set_postfix(step="Restoring Boundaries")
            # Dilate the isolated core to push the eroded edges back out
            dilated_core = ndimage.binary_dilation(core_tibia, structure=struct_element)
            
            # Bitwise AND (&): Keep the dilated pixels ONLY if they existed in the AI's original raw prediction.
            # This perfectly snaps the edges back to the exact un-eroded boundaries while leaving the fibula dead.
            cleaned_volume = (dilated_core & volume_3d).astype(np.uint8) * 255
            cleanup_pbar.update(1)

        # Step 5C: Unpack, fill and Save Back to 2D
        for z in tqdm(range(cleaned_volume.shape[0]), desc="Saving Cleaned Slices", leave=False):
            slice_2d = cleaned_volume[z]
            
            # --- Fill Cavities ---
            # Convert the slice to boolean, fill all enclosed holes, and cast back to 8-bit
            slice_boolean = slice_2d > 0
            filled_slice = ndimage.binary_fill_holes(slice_boolean)
            slice_2d = filled_slice.astype(np.uint8) * 255
            
            orig_width, orig_height = original_shapes[z]

            # --- Reverse the Proportional Padding ---
            # 1. Calculate the exact scaling factor applied during inference
            scale = 960 / max(orig_height, orig_width)
            scaled_h = int(round(orig_height * scale))
            scaled_w = int(round(orig_width * scale))
            
            # 2. Calculate Albumentations' automatic center-padding offsets
            pad_y = (960 - scaled_h) // 2
            pad_x = (960 - scaled_w) // 2
            
            # 3. Crop the artificial black borders off the predicted 960x960 slice
            cropped_slice = slice_2d[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w]
            
            # 4. Resize the cropped prediction back to the native dimensions
            final_roi = cv2.resize(cropped_slice, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            
            # 2D Median blur to smooth jagged edges without creating gray pixels
            final_roi = cv2.medianBlur(final_roi, 3)
            
            # Save strictly as 1-bit monochrome
            Image.fromarray(final_roi).convert('1').save(str(out_paths[z]))

        # Append the successfully processed folder to our tracking list
        cleaned_volumes_list.append(folder.name)    

    # Print the final success message with the dynamically injected list of cleaned volumes
    print(f"\nSuccess! All volumes cleaned in 3D and saved to {output_dir}\nCleaned Volumes: {cleaned_volumes_list}")

# =============================================================================
# --- 6. TERMINAL PARSER ---
# WHAT IT DOES: Intercepts console commands and dynamically assigns the correct 
# input, output, and model paths based on the biological target you select.
# WHY IT IS NEEDED: Drastically streamlines your workflow. It allows you to seamlessly 
# switch between processing macro whole-bone tibias or cropped trabecular VOIs directly 
# from the terminal prompt, without ever having to manually open and edit the Python script.
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and clean ROIs in 3D using trained Grayscale Attention U-Net")
    
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
        default_input = "data/inference/input_tibia_macro"
        default_output = "data/inference/output_tibia_macro"
        
    elif args.target == "cortical":
        model_weights_path = "checkpoints/tibia_cort_unet.pth"
        default_input = "data/inference/input_tibia_cort" 
        default_output = "data/inference/output_tibia_cort" 
        
    elif args.target == "trabecular":
        model_weights_path = "checkpoints/tibia_trab_unet.pth"
        default_input = "data/inference/input_tibia_trab"       
        default_output = "data/inference/output_tibia_trab" 

    # 2. Override defaults if external paths were provided in the terminal
    input_dir = args.input if args.input else default_input
    output_dir = args.output if args.output else default_output

    print(f"--- INITIALISING INFERENCE PIPELINE ---")
    print(f"Target: {args.target.upper()}")
    print(f"Model: {model_weights_path}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    generate_rois(model_weights_path, input_dir, output_dir)