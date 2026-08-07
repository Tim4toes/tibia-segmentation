# tibia-segmentation
Segment tibia and tibial cortical and trabecular bone from a microCT scan of a mouse hind leg

For black and white datasets
Currently load dataset in CTAn and use tasklist to apply threshold and save black and white image dataset.
Move black and white image dataset to the images_bw folder within the all_datasets folder.
Move the manually created ROIs (exported as .bmp with CTAn) to the mask folder within the all_datasets folder.
Run train_model_bw.py until happy with metrics.

To generate ROIs for new images, repeat same thresholding and datset saving tasklist in CTAn.
Move the new black and white datasets to the input_bw folder within the inference folder.
Run generate_rois_bw.py.
New ROIs (.bmp files) for the new black and white datasets will be stored in the output_bw folder within the inference folder.
Copy these ROIs to the original dataset folder.
Load ROI as .bmp on the original dataset in CTAn and inspect for errors.
Correct errors and save ROI as .roi file.

For grey scale images
Same as black and white, but run calculate_global_mean.py to calculate the mean and std values for the image augmentation section in train_model.py.
