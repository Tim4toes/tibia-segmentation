# tibia-segmentation
Segment tibia and tibial cortical and trabecular bone from a microCT scan of a mouse hind leg

For black and white datasets
Currently load dataset in CTAn and use tasklist to apply threshold and save black and white image dataset.
Move black and white image dataset to tibia-segmentation/data/macro_tibia/images_bw.
Move the manually created ROIs/masks (exported as .bmp with CTAn) to 
tibia-segmentation/data/macro_tibia/masks_tibia.

# How to run different training modes:
Open terminal (CMD or PowerShell) and navigate to the project directory. 
Then execute one of the following commands:

# 1 - Starting a brand-new tibia model:
python train_model_bw.py --target tibia --mode new

# 2 - Adding new datasets to your existing tibia model:
python train_model_bw.py --target tibia --mode finetune

# Resuming a crashed tibia training run:
python train_model_bw.py --target tibia --mode resume

# If you ever forget what commands are available, you can simply type:
python train_model_bw.py --help

# What the models do:
- train_model(run_mode="new", epochs=50): 
Generates a completely new random split, starts at Epoch 0, sets LR to 1e-4.

- train_model(run_mode="resume", epochs=50): 
Reads your last checkpoint, locks in the exact same validation datasets, 
loads your optimizer momentum, and picks up exactly on the epoch where you cancelled it.

- train_model(run_mode="finetune", epochs=100): 
Reads your last checkpoint, locks in the validation datasets, 
drops the LR to 1e-5, resets the epoch counter to 0, 
and begins delicate training (perfect for when you drop new datasets into your folders).

# To generate ROIs for new images, repeat same thresholding and datset saving tasklist in CTAn as for tibia.
Move the new black and white datasets to the input folderd within the inference folder (see description at end of generate_rois_bw.py).

Open terminal (CMD or Powershell)
# 1 - To segment your macro whole-bone scans, run:
python generate_rois_bw.py --target tibia

# 2 - To segment your trabecula bone scans, run:
python generate_rois_bw.py --target trabecular

# 3 - To segment your cortical bone scans, run:
python generate_rois_bw.py --target cortical

New ROIs (.bmp files) for the new black and white datasets will be stored in the output folders within the inference folder.
Copy these ROIs to the original tibia, trabecular, cortical etc. dataset folder.
Load ROI as .bmp on the original dataset in CTAn and inspect for errors.
Correct errors and save ROI as .roi file.
Run analysis in CTAn.

In future:
For grey scale images
Same as black and white, but run calculate_global_mean.py to calculate the mean and std values for the image augmentation section in train_model.py.
