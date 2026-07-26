import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from train_model import UNet # Imports your architecture

def generate_rois():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    # Check if the trained model exists before trying to load it
    model_weights_path = "tibia_unet.pth"
    if not os.path.exists(model_weights_path):
        print(f"Error: Could not find {model_weights_path}. Run train_model.py first.")
        return
        
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # Define base directories using Pathlib for easy path manipulation
    input_dir = Path("data/new_scans/")
    output_dir = Path("data/output_rois/")
    
    # Transform for inference (resizing to match training constraints)
    infer_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    print(f"Scanning for .bmp files in {input_dir}...")
    image_paths = list(input_dir.rglob("*.bmp"))
    
    if not image_paths:
        print("No .bmp files found. Please check your subfolders in new_scans.")
        return

    print(f"Generating ROIs for {len(image_paths)} images...")
    
    for img_path in image_paths:
        # 1. Reconstruct the directory structure for the output
        # E.g., data/new_scans/dataset_3/001.bmp -> dataset_3/001.bmp
        relative_path = img_path.relative_to(input_dir)
        out_path = output_dir / relative_path
        
        # Ensure the destination subfolder exists (creates it if it doesn't)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Keep the exact same filename as the input image for CT Analyser alignment
        final_out_path = out_path

        # 2. Load the original image to get its exact dimensions
        original_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        original_height, original_width = original_img.shape

        # 3. Apply transforms and move to GPU
        augmented = infer_transform(image=original_img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        # 4. Generate the prediction
        with torch.no_grad():
            prediction = model(img_tensor)
            prob_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()

        # 5. Binarize the prediction (strict 0 or 255)
        binary_mask_512 = (prob_mask > 0.5).astype(np.uint8) * 255

        # 6. Resize back to original dimensions using NEAREST to prevent grey pixels
        final_roi = cv2.resize(binary_mask_512, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # 7. Save the file
        cv2.imwrite(str(final_out_path), final_roi)
        
    print(f"Success! All {len(image_paths)} ROIs generated and mirrored seamlessly in {output_dir}")

if __name__ == "__main__":
    generate_rois()