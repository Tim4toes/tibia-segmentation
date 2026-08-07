# This script generates Regions of Interest (ROIs) for bone segmentation using a pre-trained U-Net model. 
# It processes images stored in subfolders, applies necessary transformations, and saves the resulting ROIs while maintaining the original directory structure.

import os
from tkinter import Image
from xml.parsers.expat import model
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
from train_model import UNet # Imports your architecture
from PIL import Image

def generate_rois():
    # --- MODEL INITIALISATION ---
    # Detects your RTX 3090, builds the UNet brain, and loads the weights you just 
    # trained. model.eval() locks the network so it doesn't accidentally try to 
    # learn from or alter its weights based on this new data.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    # Check if the trained model exists before trying to load it
    model_weights_path = "tibia_unet.pth"
    if not os.path.exists(model_weights_path):
        print(f"Error: Could not find {model_weights_path}. Run train_model.py first.")
        return
        
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # --- DIRECTORY AND TRANSFORM SETUP ---
    # Defines where data comes from and goes to. The infer_transform forces every 
    # incoming image to exactly 1024x1024 (matching your training size) and normalises 
    # the pixel brightness so the math behaves predictably.
    # Define base directories using Pathlib for easy path manipulation
    input_dir = Path("data/inference/input_gs/")
    output_dir = Path("data/inference/output_gs/")
    
    # Transform for inference (resizing to match training constraints)
    infer_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # --- FILE DISCOVERY ---
    # Scans the new_scans folder and all its subfolders for .bmp files.
    print(f"Scanning for .bmp files in {input_dir}...")
    image_paths = list(input_dir.rglob("*.bmp"))
    
    if not image_paths:
        print("No .bmp files found. Please check your subfolders in new_scans.")
        return

    print(f"Generating ROIs for {len(image_paths)} images...")


    # --- THE INFERENCE LOOP ---
    # Wraps the loop in tqdm so you get a live progress bar and time estimate.
    for img_path in tqdm(image_paths, desc="Processing Images"):

        # 1. Reconstruct the directory structure for the output
        # E.g., data/new_scans/dataset_3/001.bmp -> dataset_3/001.bmp
        relative_path = img_path.relative_to(input_dir)
        out_path = output_dir / relative_path
        
        # Ensure the destination subfolder exists (creates it if it doesn't)
        out_path.parent.mkdir(parents=True, exist_ok=True)
     
        # 2. Loads the raw image to capture its true, native resolution (e.g., 2752x2752)
        original_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        original_height, original_width = original_img.shape

        # 3. Apply transforms and move to GPU
        # Shrinks the image to 1024x1024 and converts it to a PyTorch tensor
        augmented = infer_transform(image=original_img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        # 4. AI Prediction
        # Uses autocast to process the math faster. The model outputs a raw probability 
        # map (values between 0 and 1) representing its confidence that a pixel is bone.
        # Generate the prediction
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                prediction = model(img_tensor)
                prob_mask_1024 = torch.sigmoid(prediction).squeeze().cpu().numpy()

        # 5. Binarize the prediction (strict 0 or 255)
        binary_mask_1024 = (prob_mask_1024 > 0.5).astype(np.uint8) * 255

        # 6. Resize back to original dimensions using NEAREST to prevent grey pixels
        final_roi_8bit = cv2.resize(binary_mask_1024, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # 7. Save as true 1-Bit Monochrome using Pillow
        # Forces the 8-bit array into strict 1-bit format so CT Analyser can read it
        final_roi_img = Image.fromarray(final_roi_8bit).convert('1')
        final_roi_img.save(str(out_path))
        
    print(f"Success! All {len(image_paths)} ROIs generated and mirrored seamlessly in {output_dir}")

if __name__ == "__main__":
    generate_rois()