# MicroCT Bone Segmentation Pipeline

This repository contains a deep learning pipeline optimized for segmenting microCT bone scans. It uses a grayscale Attention U-Net architecture to automatically isolate whole-bone macro envelopes (e.g., tibia), as well as intricate micro-architectures (cortical and trabecular bone).

The pipeline is split into two primary scripts: `train_model.py` for developing the AI, and `generate_rois.py` for deploying it. It is strictly optimized to run on a 24 GB VRAM GPU and 16 GB of system RAM, utilizing a 960x960 image resolution and restricted CPU worker limits to guarantee system stability.

---

## 1. Model Training: `train_model.py`

### What It Does

This script trains an Attention U-Net on raw grayscale `.bmp` files. It dynamically groups datasets, applies textural and spatial augmentations, and evaluates performance using a Hybrid BCE + Dice Loss function. It features an automatic learning rate scheduler and calculates HD95 distance metrics only when a new best model is saved.

### How to Use It

The script is controlled entirely via the terminal using command-line arguments. You do not need to manually edit file paths inside the Python code.

**Command Structure:**

```bash
python train_model.py --target [TARGET] --mode [MODE]

```

### Argument 1: `--target` (Required)

This argument routes the script to the correct data directories and dictates which weights file to update.

* `--target tibia`: Use this when training the macro whole-bone envelope. It looks for full-frame raw scans in the `data/macro_tibia/images` directory.
* `--target cortical`: Use this when training the dense cortical shell model. It targets cropped Volumes of Interest (VOIs) in the `data/tibia_voi/images` directory.
* `--target trabecular`: Use this when training the delicate trabecular network model. It targets the same shared cropped VOIs in the `data/tibia_voi/images` directory but uses trabecular-specific masks.

### Argument 2: `--mode` (Optional, defaults to 'new')

This argument dictates the training behavior and optimizer state.

* `--mode new`: Starts a completely brand-new training run.
* **Use Case:** When building a model from scratch. It generates a completely unseeded, random validation split, initializes a high learning rate (1e-4), and starts at Epoch 0.


* `--mode resume`: Picks up exactly where a previous run left off.
* **Use Case:** If your computer crashes or you accidentally cancel the terminal. It loads the exact same validation split, retains the optimizer's aggressive momentum, and resumes from the exact epoch it stopped on.


* `--mode finetune`: Delicately updates an established model.
* **Use Case:** When you have curated new datasets and added them to your folders. It locks in the established validation split, resets the epoch counter to 0, drops the learning rate to 1e-5, and resets momentum to zero. This integrates the new scans without erasing the model's baseline anatomical knowledge.



---

## 2. ROI Generation: `generate_rois.py`

### What It Does

This script deploys your trained models to generate Regions of Interest (ROIs) on unseen scans. It processes raw grayscale images through the network, applies a 3x3 median blur to smooth out jagged artifacts, and strictly forces the output into a 1-bit monochrome format (0 or 255) so the masks can be directly imported into software like CT Analyser for morphometry.

### How to Use It

Inference is routed dynamically via the terminal. You can use the default project folders, or seamlessly point the script to external hard drives to save time and storage space.

**Command Structure:**

```bash
python generate_rois.py --target [TARGET] [--input PATH] [--output PATH]

```

### Argument 1: `--target` (Required)

This argument determines which anatomical weights to load and establishes the default folder paths.

* `--target tibia`: Run on uncropped, full-frame microCT scans.
* `--target cortical`: Run on specific sub-regions (VOIs) pre-cropped in your analysis software.
* `--target trabecular`: Run on the exact same cropped VOIs to extract the internal strut network.

### Arguments 2 & 3: `--input` and `--output` (Optional Overrides)

By default, the script routes to the internal `data/inference/` folders. You can override these defaults to read from or write to external drives.

* **Standard Operation (Default Paths):**
Reads from and writes to the internal project folders.
```bash
python generate_rois.py --target tibia

```


* **External Input (Read from another drive):**
Reads datasets from an external folder, but saves the generated ROIs to your standard internal output folder.
```bash
python generate_rois.py --target trabecular --input "F:\External_Data\Scans\Batch_1"

```


* **External Input AND Output (Full Bypass):**
Reads from an external folder and saves the finished ROIs right back to an external drive, entirely bypassing your main project directory.
```bash
python generate_rois.py --target cortical --input "F:\External_Data\Scans\Batch_1" --output "F:\External_Data\ROIs\Batch_1_Cortical"

```



**Note on File Structure:** `generate_rois.py` will perfectly mirror whatever nested subfolder structure you place in the input directory. Ensure your incoming scans are grouped in folders (e.g., `dataset_1`, `dataset_2`) before running the script.