# Treat your entire training folder as one giant dataset. 
# You keep a running tally of two numbers across every single pixel in every single image:The sum of all pixel values ($\Sigma x$)
# The sum of all squared pixel values ($\Sigma x^2$)
# Once you have scanned every image, you calculate the global mean and standard deviation once at the very end
# Copy the two numbers it outputs, and paste them into the mean and std brackets in your augmentation blocks in train_model.py
# Repeat when training dataset is updated.

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_global_stats(image_dir):
    image_paths = list(Path(image_dir).rglob("*.bmp"))
    
    if not image_paths:
        print("No images found!")
        return

    print(f"Calculating global stats across {len(image_paths)} images...")

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for img_path in tqdm(image_paths, desc="Scanning pixels"):
        # Read the grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Scale pixels to 0.0 - 1.0 range for Albumentations
        img = img.astype(np.float32) / 255.0
        
        # Accumulate the math
        pixel_sum += np.sum(img)
        pixel_sq_sum += np.sum(img ** 2)
        pixel_count += img.size

    # Final Global Calculations
    global_mean = pixel_sum / pixel_count
    global_variance = (pixel_sq_sum / pixel_count) - (global_mean ** 2)
    global_std = np.sqrt(global_variance)

    print("\n--- CALCULATION COMPLETE ---")
    print("Paste these exact values into your Albumentations A.Normalize block:")
    print(f"mean=[{global_mean:.4f}]")
    print(f"std=[{global_std:.4f}]")

if __name__ == "__main__":
    # Point this to the main folder containing all your training images
    training_directory = "data/all_datasets/images"
    calculate_global_stats(training_directory)