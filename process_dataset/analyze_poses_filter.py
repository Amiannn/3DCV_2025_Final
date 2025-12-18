import random
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import csv
import json
from collections import defaultdict


# ---------------- CSV: angle filter table ----------------

def load_angle_filter_table(csv_path):
    """
    Load the angle filter CSV into a dict:
        { ID: [b0_45, b45_90, b90_135, b135_180] }
    where each b is a boolean.
    """
    table = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # ['ID', '0 ~ 45', '45 ~ 90', ...]

        for row in reader:
            if not row or not row[0].strip():
                # skip empty / trailing ',,,,' lines
                continue

            seq_id = row[0].strip()
            # Expect 4 columns after ID; anything missing is treated as FALSE
            flags = []
            for v in row[1:5]:
                v = (v or "").strip().upper()
                flags.append(v == "TRUE")
            # Pad to length 4 if shorter
            while len(flags) < 4:
                flags.append(False)

            table[seq_id] = flags
    return table


# ---------------- Intrinsics utilities ----------------

def load_intrinsics(intr_file):
    """
    Load intrinsics.txt into a dict:
        { frame_path: {"fx":..., "fy":..., "cx":..., "cy":..., "width":..., "height":...} }
    """
    intr = {}
    if not os.path.exists(intr_file):
        return intr

    with open(intr_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 7:
                continue

            frame_path = parts[0]
            fx, fy, cx, cy, w, h = map(float, parts[1:])

            intr[frame_path] = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": int(w),
                "height": int(h),
            }

    return intr


# ---------------- Pose utilities ----------------

def load_poses(pose_file):
    frame_paths = []
    quats = []
    translations = []

    with open(pose_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 8:
                continue

            frame_path = parts[0]
            values = list(map(float, parts[1:]))

            qw, qx, qy, qz, tx, ty, tz = values
            q = np.array([qw, qx, qy, qz], dtype=np.float64)

            if np.linalg.norm(q) < 1e-6:
                continue

            q = q / np.linalg.norm(q)

            frame_paths.append(frame_path)
            quats.append(q)
            translations.append([tx, ty, tz])

    quats = np.stack(quats, axis=0)
    translations = np.stack(translations, axis=0)
    return frame_paths, quats, translations


def quat_to_rotmat(q):
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z

    R = np.array([
        [ww + xx - yy - zz, 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     ww - xx + yy - zz, 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     ww - xx - yy + zz]
    ], dtype=np.float64)
    return R


def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to [w, x, y, z] quaternion."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-9
    return q


def compute_centers_and_dirs(quats, translations):
    """
    From world-to-camera (R, t), compute:
    - camera centers C in world coordinates
    - forward directions d in world coordinates (optical axis)
    """
    n = quats.shape[0]
    centers = np.zeros((n, 3), dtype=np.float64)
    dirs = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        R = quat_to_rotmat(quats[i])
        t = translations[i]
        # world-to-camera: x_cam = R x_world + t
        # camera center in world: C = -R^T t
        centers[i] = -R.T @ t
        # camera forward (z axis of camera) in world frame:
        dirs[i] = R.T @ np.array([0.0, 0.0, 1.0])

    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    return centers, dirs


def pairwise_rotation_angles(quats):
    dots = quats @ quats.T
    dots = np.abs(dots)
    dots = np.clip(dots, -1.0, 1.0)
    angles_rad = 2.0 * np.arccos(dots)
    return np.degrees(angles_rad)


def pairwise_baselines(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def pairwise_dir_angles(dirs):
    dots = dirs @ dirs.T
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


# ---------------- Pair selection & saving ----------------

def find_all_pairs_in_range(
    angles_rot,
    baselines,
    angles_dir,
    low,
    high,
    max_samples=None,
    max_baseline=None,
    max_dir_angle=None,
):
    """
    Return randomly-sampled pairs (i, j, angle_rot) such that:
      low <= rotation angle <= high
      baseline <= max_baseline       (if not None)
      viewing dir angle <= max_dir_angle  (if not None)

    If max_samples is None → return ALL valid pairs.
    """
    n = angles_rot.shape[0]
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            ang_r = angles_rot[i, j]

            if not (low <= ang_r <= high):
                continue

            if max_baseline is not None and baselines[i, j] > max_baseline:
                continue

            if max_dir_angle is not None and angles_dir[i, j] > max_dir_angle:
                continue

            candidates.append((i, j, ang_r))

    if max_samples is None or max_samples >= len(candidates):
        return candidates

    return random.sample(candidates, max_samples)


def save_pair_image(img1_path, img2_path, out_path):
    """
    Load two images, concatenate them horizontally, and save.
    """
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    w1, h1 = img1.size
    w2, h2 = img2.size
    h = max(h1, h2)
    w = w1 + w2

    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def relative_pose_quat(q_i, t_i, q_j, t_j):
    """
    Compute relative pose of camera j in frame of camera i,
    given world-to-camera (R_i, t_i), (R_j, t_j):

        x_j = R_rel x_i + t_rel

    Returns:
        q_rel: [w, x, y, z]
        t_rel: (3,)
    """
    R_i = quat_to_rotmat(q_i)
    R_j = quat_to_rotmat(q_j)

    R_rel = R_j @ R_i.T
    t_rel = t_j - R_j @ R_i.T @ t_i

    q_rel = rotmat_to_quat(R_rel)
    return q_rel, t_rel


def process_sequence(
    seq_id,
    angle_flags,
    root_dir,
    out_root,
    stats_list,
    max_baseline_scale=1.2,
    max_dir_angle=180.0,
):
    """
    For a given sequence ID:
      - load poses
      - load intrinsics
      - compute pairwise rotation / baseline / dir angles
      - for each angle bin that is flagged TRUE in angle_flags,
        randomly sample valid pairs and save:
          - merged pair image
          - JSON with poses relative to i (anchor) in quaternion format
          - intrinsics for i and j
      - append stats for each pair into stats_list
    """
    pose_file = os.path.join(root_dir, seq_id, "poses.txt")
    img_root_dir = os.path.join(root_dir, seq_id)
    intr_file = os.path.join(root_dir, seq_id, "intrinsics.txt")

    if not os.path.exists(pose_file):
        return

    # intrinsics dict: {frame_path: {fx, fy, cx, cy, width, height}}
    intrinsics = load_intrinsics(intr_file)

    frame_paths, quats, translations = load_poses(pose_file)
    if len(frame_paths) == 0:
        return

    angles_deg = pairwise_rotation_angles(quats)
    centers, dirs = compute_centers_and_dirs(quats, translations)
    baselines = pairwise_baselines(centers)
    dir_angles = pairwise_dir_angles(dirs)

    nonzero_baselines = baselines[baselines > 1e-6]
    if nonzero_baselines.size == 0:
        baseline_thresh = 0.0
    else:
        baseline_thresh = np.median(nonzero_baselines) * max_baseline_scale

    angle_bins = [
        (0, 45),
        (45, 90),
        (90, 135),
        (135, 180),
    ]

    for bin_idx, use_bin in enumerate(angle_flags):
        if not use_bin:
            continue

        low, high = angle_bins[bin_idx]
        bin_name = f"bin_{low}-{high}"

        pairs = find_all_pairs_in_range(
            angles_deg,
            baselines,
            dir_angles,
            low,
            high,
            max_samples=30,            # random sample up to 30
            max_baseline=baseline_thresh,
            max_dir_angle=max_dir_angle,
        )

        if not pairs:
            continue

        for k, (i, j, ang_rot) in enumerate(pairs):
            img1_rel = frame_paths[i]
            img2_rel = frame_paths[j]

            img1_path = os.path.join(img_root_dir, img1_rel)
            img2_path = os.path.join(img_root_dir, img2_rel)

            out_dir = os.path.join(out_root, seq_id, bin_name)
            os.makedirs(out_dir, exist_ok=True)

            base_name = f"{seq_id}_{bin_name}_i{i}_j{j}_rot{ang_rot:.1f}"
            img_out_path = os.path.join(out_dir, base_name + ".jpg")
            json_out_path = os.path.join(out_dir, base_name + ".json")

            # save image pair
            save_pair_image(img1_path, img2_path, img_out_path)

            # compute relative pose (i as origin)
            q_i = quats[i]
            t_i = translations[i]
            q_j = quats[j]
            t_j = translations[j]

            q_rel, t_rel = relative_pose_quat(q_i, t_i, q_j, t_j)

            rotation_deg = float(ang_rot)
            baseline_val = float(baselines[i, j])
            view_dir_angle_deg = float(dir_angles[i, j])

            data = {
                "seq_id": seq_id,
                "bin": bin_name,
                "i_index": int(i),
                "j_index": int(j),
                "image_i": img1_rel,
                "image_j": img2_rel,
                "rotation_deg": rotation_deg,
                "baseline": baseline_val,
                "view_dir_angle_deg": view_dir_angle_deg,
                "poses": {
                    "i": {
                        "q": [1.0, 0.0, 0.0, 0.0],
                        "t": [0.0, 0.0, 0.0],
                    },
                    "j": {
                        "q": [float(x) for x in q_rel],
                        "t": [float(x) for x in t_rel.tolist()],
                    },
                },
                "intrinsics": {
                    "i": intrinsics.get(img1_rel, None),
                    "j": intrinsics.get(img2_rel, None),
                },
            }

            with open(json_out_path, "w") as f:
                json.dump(data, f, indent=2)

            # collect stats
            stats_list.append(
                {
                    "seq_id": seq_id,
                    "bin": bin_name,
                    "rotation_deg": rotation_deg,
                    "baseline": baseline_val,
                    "view_dir_angle_deg": view_dir_angle_deg,
                }
            )


# ---------------- Main + statistics & plots ----------------

def main():
    root = "val/val"                        # dataset root
    csv_path = "angle_filter_table.csv"     # your CSV
    out_root = "pairs"                      # where to save pair images & json

    angle_table = load_angle_filter_table(csv_path)
    seq_ids = sorted(angle_table.keys())

    # global stats across all sequences
    stats_list = []

    for seq_id in tqdm(seq_ids):
        angle_flags = angle_table[seq_id]
        process_sequence(
            seq_id,
            angle_flags,
            root_dir=root,
            out_root=out_root,
            stats_list=stats_list,
            max_baseline_scale=1.2,
            max_dir_angle=180.0,
        )

    # ---------------- Stats summary ----------------
    if not stats_list:
        print("No pairs generated, nothing to summarize.")
        return

    rotations = np.array([s["rotation_deg"] for s in stats_list])
    baselines = np.array([s["baseline"] for s in stats_list])
    dir_angles = np.array([s["view_dir_angle_deg"] for s in stats_list])

    print("\n=== Global Pair Statistics ===")
    print(f"Total pairs: {len(stats_list)}")

    # Pairs per sequence
    pairs_per_seq = defaultdict(int)
    for s in stats_list:
        pairs_per_seq[s["seq_id"]] += 1
    print("\nPairs per sequence:")
    for seq_id, count in sorted(pairs_per_seq.items()):
        print(f"  {seq_id}: {count}")

    # Pairs per bin
    pairs_per_bin = defaultdict(int)
    for s in stats_list:
        pairs_per_bin[s["bin"]] += 1
    print("\nPairs per angle bin:")
    for bin_name, count in sorted(pairs_per_bin.items()):
        print(f"  {bin_name}: {count}")

    def print_stats(name, arr):
        print(f"\n{name}:")
        print(f"  mean   = {arr.mean():.3f}")
        print(f"  std    = {arr.std():.3f}")
        print(f"  min    = {arr.min():.3f}")
        print(f"  max    = {arr.max():.3f}")
        print(f"  median = {np.median(arr):.3f}")

    print_stats("Rotation (deg)", rotations)
    print_stats("Baseline", baselines)
    print_stats("View dir angle (deg)", dir_angles)

    # ---------------- Plots ----------------
    os.makedirs(out_root, exist_ok=True)

    # Histogram of rotation angles
    plt.figure(figsize=(6, 4))
    plt.hist(rotations, bins=36)
    plt.title("Histogram of rotation angles (deg)")
    plt.xlabel("Rotation (deg)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "pairs_stats_rotation_hist.png"))

    # Histogram of baselines
    plt.figure(figsize=(6, 4))
    plt.hist(baselines, bins=36)
    plt.title("Histogram of baselines")
    plt.xlabel("Baseline distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "pairs_stats_baseline_hist.png"))

    # Bar chart of pairs per angle bin
    plt.figure(figsize=(6, 4))
    bin_names = sorted(pairs_per_bin.keys())
    counts = [pairs_per_bin[b] for b in bin_names]
    plt.bar(bin_names, counts)
    plt.title("Number of pairs per angle bin")
    plt.xlabel("Angle bin")
    plt.ylabel("Pairs count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "pairs_stats_bin_counts.png"))

    # Optional: show plots interactively
    # plt.show()


if __name__ == "__main__":
    main()
