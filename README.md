# 3DCV 2025 Final Project

This repository contains an evaluation and visualization pipeline for **relative pose estimation** using **VGGT** and **MASt3R**.

The workflow includes:

1. Constructing a **large-viewpoint Map-free subset**
2. Running **VGGT inference**
3. Running **MASt3R inference**
4. Evaluating **relative pose errors**
5. Visualizing the results

---

## Directory Structure

```bash
.
├── evaluate
│   ├── eval_rel_pose_mast3r.py     # Evaluation script for MASt3R
│   └── eval_rel_pose_vggt.py       # Evaluation script for VGGT
├── process_dataset
│   ├── analyze_poses_filter.py     # Dataset pose analysis and subset construction
│   └── angle_filter_table.csv      # Manually curated angle filtering table
├── vggt
│   └── run.py                      # Main VGGT inference script
├── mast3r
│   ├── demo.py                     # MASt3R 3D reconstruction
│   ├── pair_test.py                # Keypoint matching
│   └── heatmap.py                  # Confidence heatmap generation
├── visual
│   └── visualize.py                # Error visualization script
└── vis.png                         # Example visualization output
```

---

## Requirements
This project **does not bundle VGGT**.
You must install them manually before running the pipeline.

* Python 3.x

* VGGT (installed as a package or available in your Python path)

* MASt3R / MEST3R (required if running MASt3R evaluation)

* Common Python packages:

  ```text
  numpy, torch, opencv-python, matplotlib, pandas
  ```

* Docker with GPU support (for MASt3R)

* Datasets and pre-trained weights can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1PY47CFyq9f9uhHuXDzEjlctS2sOuEc5W?usp=drive_link)

---

## Workflow Overview

The complete workflow has **five main steps**:

0. Construct a **large-viewpoint Map-free subset**
1. Run **VGGT inference**
2. Run **MASt3R inference**
3. Evaluate **relative pose errors**
4. Visualize **pose estimation errors**

---

## Step 0: Construct a Large-Viewpoint Map-free Subset (Required)
Before running VGGT, a **Map-free subset with large viewpoint changes** must be constructed.

```bash
python process_dataset/analyze_poses_filter.py
```

**Purpose**:

* Analyze relative pose angles in the dataset
* Filter out undesirable image pairs
* Construct a **Map-free subset with large viewpoint changes**

**Angle Filtering Table**:

```bash
process_dataset/angle_filter_table.csv
```

contains **manually annotated filtering rules**, specifying which pose angle ranges or data entries should be excluded.
This table defines what is considered **invalid or uninformative** and is used directly by `analyze_poses_filter.py`.

> In short:
>
> * `angle_filter_table.csv` = human-curated filtering rules
> * `analyze_poses_filter.py` = automated subset construction using those rules

The generated subset should be used as the **input dataset for all subsequent steps**.
---

## Step 1: Run VGGT Inference
Run VGGT on the constructed Map-free subset:
```bash
python vggt/run.py
```

This generates predicted relative poses or intermediate outputs required for evaluation.

---

## Step 2: Run MASt3R Inference

**Docker-based commands for MASt3R**:

### 2.1 3D Reconstruction (MASt3R Demo)

```bash
docker run --rm --gpus all --net=host \
  --entrypoint /bin/bash \
  -v ~/Desktop/3DCV_2025_Final/mast3r:/mast3r \
  -v ~/Desktop/3DCV_2025_Final/mast3r/docker/files/checkpoints:/mast3r/checkpoints \
  -w /mast3r \
  docker-mast3r-demo \
  -c "python3 -u demo.py --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --device cuda --local_network"
```

* Access via browser: [http://0.0.0.0:7860](http://0.0.0.0:7860)

---

### 2.2 Image Pair Keypoints Matching

```bash
docker run --rm --gpus all --net=host \
  --entrypoint /bin/bash \
  -v ~/Desktop/3DCV_2025_Final/mast3r:/mast3r \
  -v ~/Desktop/3DCV_2025_Final/mast3r/docker/files/checkpoints:/mast3r/checkpoints \
  -w /mast3r \
  docker-mast3r-demo \
  -c "python3 -u pair_test.py"
```

---

### 2.3 Generate Heatmap of Confidence

```bash
docker run --rm --gpus all --net=host \
  --entrypoint /bin/bash \
  -v ~/Desktop/3DCV_2025_Final/mast3r:/mast3r \
  -v ~/Desktop/3DCV_2025_Final/mast3r/docker/files/checkpoints:/mast3r/checkpoints \
  -w /mast3r \
  docker-mast3r-demo \
  -c "python3 heatmap.py"
```

---

## Step 3: Evaluate Relative Pose Errors

### VGGT Evaluation

```bash
python evaluate/eval_rel_pose_vggt.py
```

### MASt3R Evaluation

```bash
python eval_rel_pose.py mapfree/outputs_json_corr_v7/ \
  --output ./aggregated_stats.json \
  --plot-dir ./ \
  --pair-csv ./errors.csv
```

---

## Step 4: Visualize Errors
To visualize pose estimation errors (e.g., rotation and translation discrepancies), run:

```bash
python visual/visualize.py
```

* Shows rotation/translation discrepancies
* Helps inspect failure cases visually
* Example output: `vis.png`

---
This script helps inspect failure cases and error distributions visually.
An example output is shown in `vis.png`.
* Add **command-line argument explanations**
* Draw a **pipeline diagram** showing data flow from subset construction to visualization