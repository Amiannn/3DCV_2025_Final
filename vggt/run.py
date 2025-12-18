import os
import cv2
import json
import torch
import numpy as np
import contextlib
import matplotlib
from matplotlib import cm  # for colormap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.visual_track import get_track_colors_by_position

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fr:
        return json.load(fr)

# ============================================================
#  Heatmap (unchanged except for your latest version)
# ============================================================
def create_conf_heat_pair(
    images,
    tracks,
    conf_score,
    track_vis_mask,
    image_format="CHW",
    normalize_mode="[0,1]",
    alpha=0.5,
):
    """
    Create a side-by-side heatmap overlay for frame 0 (image_i) and frame 1 (image_j)
    using confidence scores.

    images: (S, 3, H, W) or (S, H, W, 3) on CPU
    tracks: (S, N, 2) on CPU
    conf_score: (S, N) numpy array (after squeeze)
    track_vis_mask: torch.Tensor (S, N) or None
    """

    S = images.shape[0]
    if S < 2:
        raise ValueError("Need at least 2 frames (image_i, image_j) to create a pair heatmap.")

    # Infer H, W
    if image_format == "CHW":
        H, W = images.shape[2], images.shape[3]
    else:
        H, W = images.shape[1], images.shape[2]

    def prepare_base_image(img_tensor):
        if image_format == "CHW":
            img = img_tensor.permute(1, 2, 0)  # (H, W, 3)
        else:
            img = img_tensor  # already (H, W, 3)

        img = img.numpy().astype(np.float32)

        if normalize_mode == "[0,1]":
            img = np.clip(img, 0, 1) * 255.0
        elif normalize_mode == "[-1,1]":
            img = (img + 1.0) * 0.5 * 255.0
            img = np.clip(img, 0, 255.0)
        # else: assume already roughly in [0,255]

        img = img.astype(np.uint8)
        return img  # RGB uint8

    def make_heat_overlay(frame_idx):
        # Base RGB image
        img_rgb = prepare_base_image(images[frame_idx])

        # Confidence map (single channel)
        conf_map = np.zeros((H, W), dtype=np.float32)

        pts = tracks[frame_idx].numpy()   # (N, 2)
        # you currently use the last frame's confidence for all frames
        frame_conf = conf_score[-1]  # (N,)

        if track_vis_mask is not None:
            valid_indices = torch.where(track_vis_mask[frame_idx])[0].cpu().numpy()
        else:
            valid_indices = np.arange(pts.shape[0])

        # Draw a small disk for each point instead of a single pixel
        point_radius = 3  # tweak if you want wider blobs

        for idx in valid_indices:
            x, y = pts[idx]
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                v = float(frame_conf[idx])
                if v <= 0:
                    continue
                cv2.circle(conf_map, (xi, yi), point_radius, v, thickness=-1)

        # Blur to make blobs smoother / wider
        conf_map = cv2.GaussianBlur(conf_map, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)

        # Normalize to [0,1] for colormap
        max_val = conf_map.max()
        if max_val > 0:
            norm_map = conf_map / max_val
        else:
            norm_map = conf_map

        # Apply colormap
        heat_rgba = cm.get_cmap("jet")(norm_map)  # (H, W, 4)
        heat_rgb = (heat_rgba[..., :3] * 255.0).astype(np.uint8)  # (H, W, 3)

        # Blend with original image
        base = img_rgb.astype(np.float32)
        heat = heat_rgb.astype(np.float32)
        blended = ((1.0 - alpha) * base + alpha * heat).astype(np.uint8)

        return blended

    heat_i = make_heat_overlay(0)  # image_i
    heat_j = make_heat_overlay(1)  # image_j

    pair_rgb = np.concatenate([heat_i, heat_j], axis=1)  # (H, 2W, 3)
    return pair_rgb

# ============================================================
#  Visualize tracks (unchanged)
# ============================================================
def visualize_tracks_on_images(
    images,
    tracks,
    track_vis_mask=None,
    out_dir="track_visuals_concat_by_xy",
    image_format="CHW",  # "CHW" or "HWC"
    normalize_mode="[0,1]",
    cmap_name="hsv",  # e.g. "hsv", "rainbow", "jet"
    frames_per_row=4,
    save_grid=True,
    id="",
    display_step=50,  # show only 1 point / line every `display_step` tracks
    conf_score=None,  # confidence scores for heatmap (1, S, N)
):
    """
    Visualizes frames and tracks, and optionally also creates a confidence heatmap pair.
    """

    matplotlib.use("Agg")  # for non-interactive environments

    os.makedirs(out_dir, exist_ok=True)

    # Handle batched input: (1, S, ...) -> (S, ...)
    if len(tracks.shape) == 4:
        tracks = tracks.squeeze(0)        # (S, N, 2)
        images = images.squeeze(0)        # (S, C, H, W) or (S, H, W, C)
        if track_vis_mask is not None:
            track_vis_mask = track_vis_mask.squeeze(0)  # (S, N)
        if conf_score is not None:
            conf_score = conf_score.squeeze(0)  # (S, N)

    S = images.shape[0]
    _, N, _ = tracks.shape  # (S, N, 2)

    # Move tensors to CPU
    images = images.cpu().clone()
    tracks = tracks.cpu().clone()
    if track_vis_mask is not None:
        track_vis_mask = track_vis_mask.cpu().clone()
    if conf_score is not None:
        conf_score_np = conf_score.cpu().to(torch.float32).numpy()  # (S, N)
    else:
        conf_score_np = None

    # Infer H, W from images
    if image_format == "CHW":
        H, W = images.shape[2], images.shape[3]
    else:
        H, W = images.shape[1], images.shape[2]

    # Precompute colors
    track_colors_rgb = get_track_colors_by_position(
        tracks,  # (S, N, 2)
        vis_mask_b=track_vis_mask if track_vis_mask is not None else None,
        image_width=W,
        image_height=H,
        cmap_name=cmap_name,
    )  # (N, 3), RGB in [0, 255]

    frame_images = []  # store RGB images with points

    # -------------------------------------------------------
    # Draw points per frame (subsampled by display_step)
    # -------------------------------------------------------
    for s in range(S):
        img = images[s]  # (3, H, W) or (H, W, 3)

        # Convert to (H, W, 3) RGB float
        if image_format == "CHW":
            img = img.permute(1, 2, 0)  # (H, W, 3)

        img = img.numpy().astype(np.float32)

        # Normalize to [0, 255]
        if normalize_mode == "[0,1]":
            img = np.clip(img, 0, 1) * 255.0
        elif normalize_mode == "[-1,1]":
            img = (img + 1.0) * 0.5 * 255.0
            img = np.clip(img, 0, 255.0)

        img = img.astype(np.uint8)

        # OpenCV wants BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Current frame tracks
        cur_tracks = tracks[s]  # (N, 2)
        cur_tracks_np = cur_tracks.numpy()

        if track_vis_mask is not None:
            valid_indices = torch.where(track_vis_mask[s])[0].cpu().numpy()
        else:
            valid_indices = np.arange(N)

        # Subsample by display_step for visualization
        for k in range(0, len(valid_indices), display_step):
            i = valid_indices[k]
            x, y = cur_tracks_np[i]
            pt = (int(round(x)), int(round(y)))

            R, G, B = track_colors_rgb[i]
            color_bgr = (int(B), int(G), int(R))

            cv2.circle(img_bgr, pt, radius=3, color=color_bgr, thickness=-1)

        # Back to RGB for storage
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frame_images.append(img_rgb)

    print(f"[INFO] Processed {S} frames with subsampled visualization to {out_dir}/frame_*.png")

    # -------------------------------------------------------
    # Create side-by-side images with LINES between frames
    # -------------------------------------------------------
    if S >= 2:
        for s in range(S - 1):
            left_rgb = frame_images[s].copy()
            right_rgb = frame_images[s + 1].copy()

            # Concatenate horizontally: [left | right]
            concat_rgb = np.concatenate([left_rgb, right_rgb], axis=1)  # (H, 2W, 3)

            # Tracks in left and right frames
            t0 = tracks[s].numpy()       # (N, 2)
            t1 = tracks[s + 1].numpy()   # (N, 2)

            if track_vis_mask is not None:
                vis0 = track_vis_mask[s]          # (N,)
                vis1 = track_vis_mask[s + 1]      # (N,)
                vis_both = vis0 & vis1
                valid_indices = torch.where(vis_both)[0].cpu().numpy()
            else:
                valid_indices = np.arange(N)

            # Draw lines (subsampled by display_step)
            concat_bgr = cv2.cvtColor(concat_rgb, cv2.COLOR_RGB2BGR)

            for k in range(0, len(valid_indices), display_step):
                i = valid_indices[k]
                x0, y0 = t0[i]
                x1, y1 = t1[i]

                # Shift x1 by W because right image is to the right of left image
                pt0 = (int(round(x0)), int(round(y0)))
                pt1 = (int(round(x1 + W)), int(round(y1)))

                R, G, B = track_colors_rgb[i]
                color_bgr = (int(B), int(G), int(R))

                cv2.circle(concat_bgr, pt0, radius=3, color=color_bgr, thickness=-1)
                cv2.circle(concat_bgr, pt1, radius=3, color=color_bgr, thickness=-1)
                cv2.line(concat_bgr, pt0, pt1, color=color_bgr, thickness=1)

            pair_path = os.path.join(out_dir, f"{id}_pair_{s:04d}_{s+1:04d}.png")
            cv2.imwrite(pair_path, concat_bgr)
            print(f"[INFO] Saved frame pair with lines -> {pair_path}")

    # -------------------------------------------------------
    # Confidence heatmap pair for image_i & image_j
    # -------------------------------------------------------
    if conf_score_np is not None and S >= 2:
        try:
            heat_pair = create_conf_heat_pair(
                images,
                tracks,
                conf_score_np,
                track_vis_mask,
                image_format=image_format,
                normalize_mode=normalize_mode,
                alpha=0.5,
            )
            heat_pair_path = os.path.join(out_dir, f"{id}_conf_heat_pair_0000_0001.png")
            heat_pair_bgr = cv2.cvtColor(heat_pair, cv2.COLOR_RGB2BGR)
            cv2.imwrite(heat_pair_path, heat_pair_bgr)
            print(f"[INFO] Saved confidence heatmap pair -> {heat_pair_path}")
        except Exception as e:
            print(f"[WARN] Failed to create conf heat pair: {e}")


# ============================================================
#  NEW: intrinsics & relative pose helpers
# ============================================================

### >>> NEW
def build_rescaled_K(intr_dict, H_new, W_new):
    """
    Build a 3x3 intrinsics matrix K for a resized image.

    intr_dict: {fx, fy, cx, cy, width, height}
    H_new, W_new: size of the preprocessed image tensor
    """
    width = float(intr_dict["width"])
    height = float(intr_dict["height"])

    sx = W_new / width
    sy = H_new / height

    fx = intr_dict["fx"] * sx
    fy = intr_dict["fy"] * sy
    cx = intr_dict["cx"] * sx
    cy = intr_dict["cy"] * sy

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


### >>> NEW
def compute_relative_pose_vggt(extrinsic):
    """
    Compute relative pose (R_ji, t_ji) from VGGT extrinsics.

    extrinsic: torch.Tensor (1, S, 3, 4), camera-from-world [R|t]
    We assume images[0] = image_i, images[1] = image_j.
    Returns R_ji, t_ji as numpy arrays, where j <- i.
    """
    ext = extrinsic.squeeze(0).cpu().numpy()  # (S, 3, 4)
    assert ext.shape[0] >= 2, "Need at least 2 frames for relative pose."

    R_i = ext[0, :, :3]   # (3, 3)
    t_i = ext[0, :, 3]    # (3,)
    R_j = ext[1, :, :3]
    t_j = ext[1, :, 3]

    # camera-from-world: X_cam = R * X_world + t
    # Relative transform from cam i to cam j:
    # R_ji = R_j * R_i^T
    # t_ji = t_j - R_ji * t_i
    R_ji = R_j @ R_i.T
    t_ji = t_j - R_ji @ t_i

    return R_ji, t_ji


### >>> NEW
def compute_relative_pose_from_tracks(
    track,
    track_vis_mask,
    K_i,
    K_j,
    min_points=30,
):
    """
    Estimate relative pose from 2D correspondences + intrinsics.

    track: torch.Tensor (1, S, N, 2)
    track_vis_mask: torch.Tensor (1, S, N) (bool)
    K_i, K_j: 3x3 intrinsics matrices for image_i, image_j
    Returns (R_ji, t_ji) from image_i to image_j (up to scale), or (None, None) if fail.
    """

    track_cpu = track.squeeze(0).cpu()         # (S, N, 2)
    mask_cpu = track_vis_mask.squeeze(0).cpu() # (S, N)
    S, N, _ = track_cpu.shape
    assert S >= 2, "Need at least 2 frames."

    pts_i = track_cpu[0]  # (N, 2)
    pts_j = track_cpu[1]  # (N, 2)

    vis_i = mask_cpu[0].bool()
    vis_j = mask_cpu[1].bool()
    valid = (vis_i & vis_j).numpy()

    pts_i = pts_i[valid].numpy()  # (M, 2)
    pts_j = pts_j[valid].numpy()  # (M, 2)

    M = pts_i.shape[0]
    print(f"[POSE-TRACK] #valid correspondences: {M}")
    if M < min_points:
        print("[POSE-TRACK] Not enough correspondences, skipping pose-from-tracks.")
        return None, None

    # Normalize using intrinsics
    pts_i_norm = cv2.undistortPoints(
        pts_i.reshape(-1, 1, 2),
        cameraMatrix=K_i,
        distCoeffs=None
    ).reshape(-1, 2)
    pts_j_norm = cv2.undistortPoints(
        pts_j.reshape(-1, 1, 2),
        cameraMatrix=K_j,
        distCoeffs=None
    ).reshape(-1, 2)

    # Estimate Essential matrix
    E, inliers = cv2.findEssentialMat(
        pts_i_norm,
        pts_j_norm,
        method=cv2.RANSAC,
        threshold=1e-3,
        prob=0.999
    )

    if E is None:
        print("[POSE-TRACK] findEssentialMat failed.")
        return None, None

    # Recover relative pose (j <- i)
    _, R_ji, t_ji, mask_pose = cv2.recoverPose(E, pts_i_norm, pts_j_norm)

    t_ji = t_ji.reshape(3)  # (3,)

    return R_ji, t_ji


# ============================================================
#  Query points
# ============================================================
def create_full_pixel_query_points(images, device, pixel_step=4):
    """
    Create a subsampled pixel grid.
    pixel_step: how many pixels to skip (e.g., 4 → sample every 4 pixels)
    """
    _, _, _, H, W = images.shape

    xs = torch.arange(0, W, pixel_step)
    ys = torch.arange(0, H, pixel_step)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # (nx, ny)

    # (N, 2) in (x, y)
    query_points = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1
    ).to(device=device, dtype=torch.float32)

    print(f"Created {query_points.shape[0]} subsampled query points with pixel_step={pixel_step}")
    return query_points


# ============================================================
#  Main per-pair run
# ============================================================
def run(image_names, model, dtype, device, id, datas):
    images = load_and_preprocess_images(image_names).to(device)  # (S, 3, H, W)

    # Remember preprocessed H, W for intrinsics rescaling
    S, C, H, W = images.shape

    # Autocast context (only on CUDA)
    if device == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            # Add batch dimension: (1, S, 3, H, W)
            images = images[None]

            # Aggregator
            aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )
            # extrinsic: (1, S, 3, 4), intrinsic: (1, S, 3, 3)

            # =====================================================
            # 1) Relative pose from VGGT camera head
            # =====================================================
            R_ji_vggt, t_ji_vggt = compute_relative_pose_vggt(extrinsic)
            print("\n[POSE-VGGT] Relative pose j <- i (R, t):")
            print("R_ji_vggt =\n", R_ji_vggt)
            print("t_ji_vggt =", t_ji_vggt)

            # Predict Depth Maps (not strictly needed for pose)
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            # ------------------------------------------------------------------
            # Create query points on a subsampled grid (memory-friendly)
            # ------------------------------------------------------------------
            query_points = create_full_pixel_query_points(images, device, pixel_step=4)
            # ------------------------------------------------------------------

            # Predict Tracks
            track_list, vis_score, conf_score = model.track_head(
                aggregated_tokens_list,
                images,
                ps_idx,
                query_points=query_points[None],  # (1, N, 2)
            )

            print("Track list shapes (per scale):", [t.shape for t in track_list])
            print("Visibility score shape:", vis_score.shape)
            print("Confidence score shape:", conf_score.shape)

            # Take the last (highest-res) level of tracks
            track = track_list[-1]  # (1, S, N, 2)

            # Visibility mask using both visibility and confidence
            track_vis_mask = (conf_score > 0.1) & (vis_score > 0.1)  # (1, S, N)

            # =====================================================
            # 2) Relative pose from correspondences + intrinsics
            # =====================================================
            intr_i = datas["intrinsics"]["i"]
            intr_j = datas["intrinsics"]["j"]

            K_i = build_rescaled_K(intr_i, H, W)
            K_j = build_rescaled_K(intr_j, H, W)

            R_ji_corr, t_ji_corr = compute_relative_pose_from_tracks(
                track,
                track_vis_mask,
                K_i,
                K_j,
                min_points=30,
            )

            if R_ji_corr is not None:
                print("\n[POSE-TRACK] Relative pose j <- i from tracks (R, t) (up to scale):")
                print("R_ji_corr =\n", R_ji_corr)
                print("t_ji_corr =", t_ji_corr)
            else:
                print("\n[POSE-TRACK] Failed to estimate pose from correspondences.")

            # =====================================================
            # Save all pose results to JSON
            # =====================================================
            save_path = os.path.join("track_visuals_v2", f"{id}_pose_results.json")

            # ground truth
            gt_i_q = datas["poses"]["i"]["q"]
            gt_i_t = datas["poses"]["i"]["t"]
            gt_j_q = datas["poses"]["j"]["q"]
            gt_j_t = datas["poses"]["j"]["t"]

            def quat_to_R(q):
                # Convert quaternion [w,x,y,z] to rotation matrix
                w, x, y, z = q
                R = np.array([
                    [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
                    [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
                    [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
                ], dtype=float)
                return R

            R_i_gt = quat_to_R(gt_i_q)
            R_j_gt = quat_to_R(gt_j_q)
            t_i_gt = np.array(gt_i_t, dtype=float)
            t_j_gt = np.array(gt_j_t, dtype=float)

            # Compute GT relative pose j <- i
            R_ji_gt = R_j_gt @ R_i_gt.T
            t_ji_gt = t_j_gt - R_ji_gt @ t_i_gt

            # Build result dict
            result = {
                "id": id,
                "intrinsics": datas["intrinsics"],

                "gt_pose": {
                    "i": {"R": R_i_gt.tolist(), "t": t_i_gt.tolist()},
                    "j": {"R": R_j_gt.tolist(), "t": t_j_gt.tolist()},
                    "R_ji_gt": R_ji_gt.tolist(),
                    "t_ji_gt": t_ji_gt.tolist(),
                },

                "vggt_pose": {
                    "R_ji": R_ji_vggt.tolist(),
                    "t_ji": t_ji_vggt.tolist()
                },

                "corr_pose": {
                    "R_ji": R_ji_corr.tolist() if R_ji_corr is not None else None,
                    "t_ji": t_ji_corr.tolist() if R_ji_corr is not None else None,
                    "num_inliers": int(np.sum(track_vis_mask[0,0].cpu().numpy()))
                }
            }

            # Write JSON
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"[INFO] Saved pose results → {save_path}")

            # =====================================================
            # Visualization (as before)
            # =====================================================
            out_dir = "track_visuals_v2"
            os.makedirs(out_dir, exist_ok=True)

            visualize_tracks_on_images(
                images,                 # (1, S, 3, H, W)
                track,                  # (1, S, N, 2)
                track_vis_mask=track_vis_mask,
                out_dir=out_dir,
                frames_per_row=2,
                image_format="CHW",     # images are (C, H, W)
                normalize_mode="[0,1]", # load_and_preprocess_images gives [0,1] range
                id=id,
                display_step=150,
                conf_score=conf_score,
            )

            print(f"Saved visualized tracks to folder: {out_dir}")


# ============================================================
#  Entry point
# ============================================================
def main():
    # Select device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print(f"Using device: {device}, dtype: {dtype}")

    # Initialize model (will download weights on first run)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    root_folder = "/media/ai2lab/storage1/experiments/mapfree/pairs"
    image_root_path = "/media/ai2lab/storage1/experiments/mapfree/val/val"

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            for filename in os.listdir(sub_folder_path):
                if '.json' not in filename:
                    continue
                data_path = os.path.join(sub_folder_path, filename)
                datas = read_json(data_path)

                seq_id  = datas['seq_id']
                image_i = os.path.join(image_root_path, seq_id, datas['image_i'])
                image_j = os.path.join(image_root_path, seq_id, datas['image_j'])

                image_names = [image_i, image_j]
                assert all(os.path.exists(p) for p in image_names), "One or more image paths do not exist."

                # >>> pass datas into run so we can use intrinsics
                run(image_names, model, dtype, device, filename, datas)


if __name__ == "__main__":
    main()
