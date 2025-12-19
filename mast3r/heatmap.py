import os
import json
import torch
import numpy as np
import tempfile
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from PIL import Image
import sys
import cv2
import matplotlib
from matplotlib import cm

# --- Add dust3r to path ---
sys.path.append('./dust3r') 

try:
    from mast3r.model import AsymmetricMASt3R
except ImportError:
    from dust3r.model import AsymmetricMASt3R

from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from mast3r.fast_nn import fast_reciprocal_NNs 

# ==========================================
# [Configuration]
# ==========================================

PAIRS_ROOT = "./3DCV/mapfree/pairs"
OUTPUT_DIR = "./3DCV/mapfree/outputs_json_corr_v7"     
VIS_DIR = "./3DCV/mapfree/outputs_vis_v7"

MODEL_PATH = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OVERWRITE = True 
MATCH_THRESHOLD = 0
MIN_MATCHES = 30 

# ==========================================
# [Safety Helpers]
# ==========================================

def recursive_to_numpy(obj):
    """
    Recursively convert torch.Tensor (on any device) to numpy arrays.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, list):
        return [recursive_to_numpy(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_numpy(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj
    return obj

def get_track_colors_by_position(tracks, image_width=512, image_height=512, cmap_name="hsv"):
    # tracks: (S, N, 2) numpy
    pts = tracks[0] # (N, 2)
    
    w = max(float(image_width), 1.0)
    h = max(float(image_height), 1.0)
    
    x = pts[:, 0] / w
    y = pts[:, 1] / h
    
    cx, cy = 0.5, 0.5
    dx = x - cx
    dy = y - cy
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) # [0, 1]
    
    colormap = cm.get_cmap(cmap_name)
    colors = colormap(angle) # (N, 4)
    
    colors_rgb = (colors[:, :3] * 255).astype(np.uint8)
    return colors_rgb

def create_conf_heat_pair(images_np, tracks_np, conf_np, track_vis_mask, image_format="CHW", normalize_mode="[0,1]", alpha=0.5):
    # All inputs assumed to be CPU Numpy by now
    S = images_np.shape[0]
    if image_format == "CHW":
        H, W = images_np.shape[2], images_np.shape[3]
    else:
        H, W = images_np.shape[1], images_np.shape[2]

    def prepare_base_image(img_arr):
        if image_format == "CHW":
            img = np.transpose(img_arr, (1, 2, 0)) # HWC
        else:
            img = img_arr
            
        if normalize_mode == "[0,1]":
            img = np.clip(img, 0, 1) * 255.0
        elif normalize_mode == "[-1,1]":
            img = (img + 1.0) * 0.5 * 255.0
            img = np.clip(img, 0, 255.0)
        return img.astype(np.uint8)

    def make_heat_overlay(frame_idx):
        img_rgb = prepare_base_image(images_np[frame_idx])
        conf_map = np.zeros((H, W), dtype=np.float32)
        pts = tracks_np[frame_idx]
        
        valid_indices = np.arange(pts.shape[0])
        
        point_radius = 6
        for idx in valid_indices:
            x, y = pts[idx]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                v = float(conf_np[idx]) if conf_np is not None else 1.0
                if v <= 0: continue
                cv2.circle(conf_map, (xi, yi), point_radius, v, thickness=-1)

        conf_map = cv2.GaussianBlur(conf_map, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)
        max_val = conf_map.max()
        norm_map = conf_map / max_val if max_val > 0 else conf_map
        
        heat_rgba = cm.get_cmap("jet")(norm_map)
        heat_rgb = (heat_rgba[..., :3] * 255.0).astype(np.uint8)
        
        base = img_rgb.astype(np.float32)
        heat = heat_rgb.astype(np.float32)
        blended = ((1.0 - alpha) * base + alpha * heat).astype(np.uint8)
        return blended

    heat_i = make_heat_overlay(0)
    heat_j = make_heat_overlay(1)
    pair_rgb = np.concatenate([heat_i, heat_j], axis=1)
    return pair_rgb
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)
def visualize_tracks_on_images(images, tracks, conf_score=None, out_dir=".", id="", display_step=50):
    # Guaranteed CPU Numpy inputs via recursive_to_numpy calls in main loop
    images_np = images
    tracks_np = tracks
    conf_score_np = conf_score
    
    S = images_np.shape[0]
    _, N, _ = tracks_np.shape
    H, W = images_np.shape[2], images_np.shape[3] # Assume CHW

    track_colors_rgb = get_track_colors_by_position(tracks_np, image_width=W, image_height=H)
    
    frame_images = []
    
    # 1. Dots
    for s in range(S):
        img = np.transpose(images_np[s], (1, 2, 0)) # HWC
        img = np.clip(img, 0, 1) * 255.0
        img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        cur_tracks = tracks_np[s]
        
        for k in range(0, N, display_step):
            x, y = cur_tracks[k]
            pt = (int(round(x)), int(round(y)))
            R, G, B = track_colors_rgb[k]
            cv2.circle(img_bgr, pt, radius=3, color=(int(B), int(G), int(R)), thickness=-1)
            
        frame_images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # 2. Pair Lines
    left_rgb = frame_images[0].copy()
    right_rgb = frame_images[1].copy()
    concat_rgb = np.concatenate([left_rgb, right_rgb], axis=1)
    concat_bgr = cv2.cvtColor(concat_rgb, cv2.COLOR_RGB2BGR)
    
    t0 = tracks_np[0]
    t1 = tracks_np[1]
    
    for k in range(0, N, display_step):
        x0, y0 = t0[k]
        x1, y1 = t1[k]
        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1 + W)), int(round(y1)))
        R, G, B = track_colors_rgb[k]
        color_bgr = (int(B), int(G), int(R))
        
        cv2.circle(concat_bgr, pt0, radius=3, color=color_bgr, thickness=-1)
        cv2.circle(concat_bgr, pt1, radius=3, color=color_bgr, thickness=-1)
        cv2.line(concat_bgr, pt0, pt1, color=color_bgr, thickness=1)

    pair_path = os.path.join(out_dir, f"{id}_pair.png")
    cv2.imwrite(pair_path, concat_bgr)
    
    # 3. Heatmap
    if conf_score_np is not None:
        try:
            heat_pair = create_conf_heat_pair(images_np, tracks_np, conf_score_np, None, image_format="CHW")
            heat_path = os.path.join(out_dir, f"{id}_conf.png")
            cv2.imwrite(heat_path, cv2.cvtColor(heat_pair, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Heatmap error: {e}")

# ==========================================
# [VGGT Helpers]
# ==========================================

def build_rescaled_K(intr_dict, H_new, W_new):
    width = float(intr_dict["width"])
    height = float(intr_dict["height"])
    sx = W_new / width
    sy = H_new / height
    fx = intr_dict["fx"] * sx
    fy = intr_dict["fy"] * sy
    cx = intr_dict["cx"] * sx
    cy = intr_dict["cy"] * sy
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def compute_relative_pose_from_tracks(pts_i, pts_j, K_i, K_j, min_points=30):
    if pts_i.shape[0] < min_points:
        return None, None

    pts_i = np.ascontiguousarray(pts_i, dtype=np.float32)
    pts_j = np.ascontiguousarray(pts_j, dtype=np.float32)

    pts_i_norm = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), cameraMatrix=K_i, distCoeffs=None).reshape(-1, 2)
    pts_j_norm = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), cameraMatrix=K_j, distCoeffs=None).reshape(-1, 2)

    E, inliers = cv2.findEssentialMat(pts_i_norm, pts_j_norm, method=cv2.RANSAC, threshold=1e-3, prob=0.999)
    if E is None: return None, None
    _, R_ji, t_ji, mask_pose = cv2.recoverPose(E, pts_i_norm, pts_j_norm)
    return R_ji, t_ji.reshape(3)

def q_t_to_matrix(q, t):
    q_scipy = [q[1], q[2], q[3], q[0]] 
    R = Rotation.from_quat(q_scipy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_relative_pose_gt(T_i_w, T_j_w):
    # Ensure inputs are numpy arrays (doubly safe)
    if isinstance(T_i_w, torch.Tensor): T_i_w = T_i_w.detach().cpu().numpy()
    if isinstance(T_j_w, torch.Tensor): T_j_w = T_j_w.detach().cpu().numpy()
    
    T_j_inv = np.linalg.inv(T_j_w)
    T_ji = T_j_inv @ T_i_w
    return T_ji[:3, :3], T_ji[:3, 3]

def load_split_images(json_path):
    sbs_path = str(json_path).replace(".json", ".jpg")
    if not os.path.exists(sbs_path): return None, None
    try:
        img_sbs = Image.open(sbs_path).convert('RGB')
        w, h = img_sbs.size
        img_i = img_sbs.crop((0, 0, w//2, h))
        img_j = img_sbs.crop((w//2, 0, w, h))
        return img_i, img_j
    except: return None, None

# ==========================================
# Main Logic
# ==========================================

def process_single_pair(json_file, model):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 1. GT Pose
            pose_i, pose_j = data["poses"]["i"], data["poses"]["j"]
            T_i = q_t_to_matrix(pose_i["q"], pose_i["t"])
            T_j = q_t_to_matrix(pose_j["q"], pose_j["t"])
            R_gt, t_gt = get_relative_pose_gt(T_i, T_j)

            # 2. Images
            pil_i, pil_j = load_split_images(json_file)
            if pil_i is None: return None
            path_i, path_j = os.path.join(temp_dir, "i.jpg"), os.path.join(temp_dir, "j.jpg")
            pil_i.save(path_i); pil_j.save(path_j)
            
            imgs = load_images([path_i, path_j], size=512, verbose=False)
            pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)
            
            # Helper to ensure tensor
            def ensure_tensor_on_device(d, device):
                new_d = {}
                for k, v in d.items():
                    if isinstance(v, np.ndarray):
                        if v.dtype.type is np.str_ or v.dtype.type is np.object_: pass 
                        else: v = torch.from_numpy(v)
                    if isinstance(v, torch.Tensor):
                        new_d[k] = v.to(device)
                    else:
                        new_d[k] = v
                return new_d

            view1_gpu = ensure_tensor_on_device(pairs[0][0], DEVICE)
            view2_gpu = ensure_tensor_on_device(pairs[0][1], DEVICE)

            # 3. MASt3R Inference
            with torch.no_grad():
                preds = model(view1_gpu, view2_gpu)

            # 4. Extract Matches
            desc1 = preds[0]['desc'].squeeze(0).detach()
            desc2 = preds[1]['desc'].squeeze(0).detach()
            
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=4, device=DEVICE, dist='dot', block_size=2**13)
            
            feats1 = desc1[matches_im0[:, 1], matches_im0[:, 0]]
            feats2 = desc2[matches_im1[:, 1], matches_im1[:, 0]]
            scores = (feats1 * feats2).sum(dim=-1)

            # =========================================================
            # [Step 4b] FORCE CPU NUMPY CONVERSION IMMEDIATELY
            # =========================================================
            # This eliminates ANY chance of mixing GPU/CPU during logic
            H, W = view1_gpu['true_shape'][0].detach().cpu().numpy().astype(int).tolist()
            
            m0_np = to_numpy(matches_im0)
            m1_np = to_numpy(matches_im1)
            sc_np = to_numpy(scores)
            
            # Pure Numpy Logic (Safe from CUDA errors)
            valid_mask = (sc_np > MATCH_THRESHOLD) & \
                         (m0_np[:, 0] < W) & (m0_np[:, 1] < H) & \
                         (m1_np[:, 0] < W) & (m1_np[:, 1] < H)
            
            m0 = m0_np[valid_mask]
            m1 = m1_np[valid_mask]
            sc = sc_np[valid_mask]
            
            if len(m0) == 0:
                return None

            # [Step 4c] Visualization
            try:
                # Prepare Inputs: explicitly use CPU Tensors or Numpy
                img1_norm = (view1_gpu['img'] + 1.0) * 0.5
                img2_norm = (view2_gpu['img'] + 1.0) * 0.5
                
                # Convert images to CPU Numpy before concatenation
                img1_np = recursive_to_numpy(img1_norm)
                img2_np = recursive_to_numpy(img2_norm)
                
                images_viz = np.concatenate([img1_np, img2_np], axis=0) # (2, 3, H, W)
                tracks_viz = np.stack([m0, m1], axis=0) # (2, N, 2) Numpy

                vis_id = f"{data['seq_id']}_{json_file.stem}"
                visualize_tracks_on_images(
                    images_viz, 
                    tracks_viz, 
                    conf_score=sc, 
                    out_dir=VIS_DIR, 
                    id=vis_id, 
                    display_step=20 
                )
            except Exception as e_vis:
                print(f"[Vis Error] {vis_id}: {e_vis}")

            # 5. Corr Pose
            intr_i, intr_j = data["intrinsics"]["i"], data["intrinsics"]["j"]
            K_i = build_rescaled_K(intr_i, H, W)
            K_j = build_rescaled_K(intr_j, H, W)
            R_corr, t_corr = compute_relative_pose_from_tracks(m0, m1, K_i, K_j, min_points=MIN_MATCHES)

            # 6. Master Pose
            optim_cache = os.path.join(temp_dir, "optim")
            os.makedirs(optim_cache, exist_ok=True)
            scene = sparse_global_alignment([path_i, path_j], pairs, optim_cache, model, lr1=0.07, niter1=300, lr2=0.01, niter2=200, device=DEVICE, opt_depth=True, verbose=False)
            
            poses_est_raw = scene.get_im_poses()
            poses_est = recursive_to_numpy(poses_est_raw) 
            
            R_mas, t_mas = get_relative_pose_gt(poses_est[0], poses_est[1])

            del scene, pairs, imgs, preds, view1_gpu, view2_gpu
            torch.cuda.empty_cache()

            vis_id = f"{data['seq_id']}_{json_file.stem}" 
            return {
                "id": vis_id,
                "intrinsics": data["intrinsics"],
                "heatmap_path": os.path.join(VIS_DIR, f"{vis_id}_conf.png"),
                "pair_path": os.path.join(VIS_DIR, f"{vis_id}_pair.png"),
                "gt_pose": {"R_ji_gt": R_gt.tolist(), "t_ji_gt": t_gt.tolist()},
                "master_pose": {"R_ji": R_mas.tolist(), "t_ji": t_mas.tolist()},
                "corr_pose": {"R_ji": R_corr.tolist() if R_corr is not None else None, "t_ji": t_corr.tolist() if R_corr is not None else None, "num_inliers": int(len(m0))}
            }

        except Exception as e:
            print(f"Error {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print(f"Loading MASt3R ({DEVICE})...")
    model = AsymmetricMASt3R.from_pretrained(MODEL_PATH).to(DEVICE)

    root_path = Path(PAIRS_ROOT)
    json_files = list(root_path.rglob("s*_i*_j*.json"))
    print(f"Found {len(json_files)} pairs.")

    for jf in tqdm(json_files, desc="Processing"):
        out_path = os.path.join(OUTPUT_DIR, jf.name + "_res.json")
        if not OVERWRITE and os.path.exists(out_path): continue
        
        res = process_single_pair(jf, model)
        if res:
            with open(out_path, 'w') as f: json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()