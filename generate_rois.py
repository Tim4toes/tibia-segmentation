import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train_model import UNet # Imports your architecture

def generate_rois():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load("tibia_unet.pth"))
    model.eval()

    input_dir = "data/new_scans/"
    output_dir = "data/output_rois/"
    os.makedirs(output_dir, exist_ok=True)

    # Transform for inference (resizing to match training)
    infer_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    print("Generating ROIs...")
    for filename in os.listdir(input_dir):
        if not filename.endswith(".bmp"):
            continue
            
        img_path = os.path.join(input_dir, filename)
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        original_height, original_width = original_img.shape

        # Apply transforms
        augmented = infer_transform(image=original_img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)
            prob_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()

        # Binarize the prediction
        binary_mask_512 = (prob_mask > 0.5).astype(np.uint8) * 255

        # Resize back to original dimensions using NEAREST to prevent grey pixels
        final_roi = cv2.resize(binary_mask_512, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        save_path = os.path.join(output_dir, filename.replace('.bmp', '_ROI.bmp'))
        cv2.imwrite(save_path, final_roi)
        
    print("All ROIs generated and ready for CT Analyser!")

if __name__ == "__main__":
    generate_rois()