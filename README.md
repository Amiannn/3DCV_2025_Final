# 3DCV 2025 Final Project

This repository contains an evaluation and visualization pipeline for relative pose estimation using **VGGT** and **MASt3R**.
The workflow includes **constructing a large-viewpoint Map-free subset**, running VGGT, MASt3R inference, evaluating relative pose errors, and visualizing the results.

---

## Directory Structure

```bash
.
├── evaluate
│   ├── eval_rel_pose_mast3r.py     # Evaluation script for MASt3R (optional)
│   └── eval_rel_pose_vggt.py       # Evaluation script for VGGT
├── process_dataset
│   ├── analyze_poses_filter.py     # Dataset pose analysis and subset construction
│   └── angle_filter_table.csv      # Manually curated angle filtering table
├── vggt
│   └── run.py                      # Main VGGT inference script
├── vis.png                         # Example visualization output
└── visual
    └── visualize.py                # Error visualization script
```

---

## Requirements

This project **does not bundle VGGT and MEST3R**.
You must install them manually before running the pipeline.

### Required Packages

* Python 3.x
* VGGT (installed as a package or available in your Python path)
* MASt3R / MEST3R (only required if running MASt3R evaluation)
* Common scientific libraries:

  * numpy
  * torch
  * opencv-python
  * matplotlib
  * pandas

---

## Workflow Overview

The complete workflow follows **four steps**:

0. **Construct a large-viewpoint Map-free subset**
1. **Run VGGT inference**
2. **Evaluate relative pose errors**
3. **Visualize errors**

---

## Step 0: Construct a Large-Viewpoint Map-free Subset (Required)

Before running VGGT, a **Map-free subset with large viewpoint changes** must be constructed.

This is done using:

```bash
python process_dataset/analyze_poses_filter.py
```

### Purpose

* Analyze relative pose angles in the dataset
* Filter out undesirable image pairs
* Construct a **Map-free subset with large viewpoint changes**

### Angle Filtering Table

The file:

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

This step generates predicted relative poses or intermediate outputs required for evaluation.

---

## Step 2: Evaluate Relative Pose (VGGT)

After inference, evaluate relative pose errors using:

```bash
python evaluate/eval_rel_pose_vggt.py
```

This script computes relative rotation and translation errors based on VGGT outputs and ground truth poses.

> To evaluate MASt3R instead, run:
>
> ```bash
> python evaluate/eval_rel_pose_mast3r.py
> ```

---

## Step 3: Visualize Errors

To visualize pose estimation errors (e.g., rotation and translation discrepancies), run:

```bash
python visual/visualize.py
```

This script helps inspect failure cases and error distributions visually.
An example output is shown in `vis.png`.
* Add **command-line argument explanations**
* Draw a **pipeline diagram** showing data flow from subset construction to visualization
