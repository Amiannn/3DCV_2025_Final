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
import matplotlib.pyplot as pl
from matplotlib.collections import LineCollection

# --- Add dust3r to path ---
sys.path.append('./dust3r') 

# ==========================================
# Imports
# ==========================================
try:
    from mast3r.model import AsymmetricMASt3R
except ImportError:
    from dust3r.model import AsymmetricMASt3R

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
# [新增] 用於計算匹配分數的模組
from mast3r.fast_nn import fast_reciprocal_NNs 

# ==========================================
# [Configuration]
# ==========================================

PAIRS_ROOT = "./3DCV/mapfree/pairs"
OUTPUT_DIR = "./3DCV/mapfree/outputs_json"
HEATMAP_DIR = "./3DCV/mapfree/outputs_heatmap" 

MODEL_PATH = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization Settings
MATCH_THRESHOLD = 0.20
LINE_ALPHA = 0.3

# ==========================================
# Helper Functions
# ==========================================

def q_t_to_matrix(q, t):
    q_scipy = [q[1], q[2], q[3], q[0]] 
    R = Rotation.from_quat(q_scipy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_relative_pose(T_i_w, T_j_w):
    T_j_inv = np.linalg.inv(T_j_w)
    T_ji = T_j_inv @ T_i_w
    return T_ji[:3, :3], T_ji[:3, 3]

def load_split_images(json_path):
    sbs_path = str(json_path).replace(".json", ".jpg")
    if not os.path.exists(sbs_path):
        return None, None

    try:
        img_sbs = Image.open(sbs_path).convert('RGB')
        w, h = img_sbs.size
        img_i = img_sbs.crop((0, 0, w//2, h))
        img_j = img_sbs.crop((w//2, 0, w, h))
        return img_i, img_j
    except Exception as e:
        print(f"[Warn] Error reading image {sbs_path}: {e}")
        return None, None

def save_matches_visualization(view1, view2, preds, save_path):
    """
    整合了使用者提供的 Reference Code 邏輯：
    使用 fast_reciprocal_NNs 計算匹配並繪製連線圖。
    """
    try:
        # 1. 提取 Descriptor (Batch size=1, so we squeeze)
        # preds 是 tuple (dict1, dict2)
        desc1 = preds[0]['desc'].squeeze(0).detach()
        desc2 = preds[1]['desc'].squeeze(0).detach()
        
        # 2. 尋找 Reciprocal Nearest Neighbors
        # subsample_or_initxy1=8 是為了加速，也可以設為 1 獲取最密集的點
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, 
            subsample_or_initxy1=8,
            device=DEVICE, 
            dist='dot', 
            block_size=2**13
        )

        # 3. 計算分數 (Dot Product)
        feats1 = desc1[matches_im0[:, 1], matches_im0[:, 0]]
        feats2 = desc2[matches_im1[:, 1], matches_im1[:, 0]]
        scores = (feats1 * feats2).sum(dim=-1)

        # 4. 處理邊界過濾
        H0, W0 = view1['true_shape'][0].cpu().numpy()
        H1, W1 = view2['true_shape'][0].cpu().numpy()

        # 確保轉為 Numpy 進行邏輯運算
        m0_x, m0_y = matches_im0[:, 0].cpu().numpy(), matches_im0[:, 1].cpu().numpy()
        m1_x, m1_y = matches_im1[:, 0].cpu().numpy(), matches_im1[:, 1].cpu().numpy()

        valid_border_0 = (m0_x >= 3) & (m0_x < int(W0) - 3) & (m0_y >= 3) & (m0_y < int(H0) - 3)
        valid_border_1 = (m1_x >= 3) & (m1_x < int(W1) - 3) & (m1_y >= 3) & (m1_y < int(H1) - 3)
        
        valid_score = (scores >= MATCH_THRESHOLD).cpu().numpy()
        
        final_mask = valid_border_0 & valid_border_1 & valid_score

        # 應用 Mask
        matches_im0 = matches_im0.cpu().numpy()[final_mask]
        matches_im1 = matches_im1.cpu().numpy()[final_mask]
        final_scores = scores.cpu().numpy()[final_mask]

        if len(matches_im0) == 0:
            print(f"[Vis Info] No matches found for {save_path.name}")
            return

        # 5. 繪圖 (使用 Matplotlib)
        # 準備背景圖
        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        def get_rgb_np(view):
            rgb = view['img'].cpu() * image_std + image_mean
            return rgb.squeeze(0).permute(1, 2, 0).numpy()

        img0_np = get_rgb_np(view1)
        img1_np = get_rgb_np(view2)

        # 拼合畫布
        H0, W0, _ = img0_np.shape
        H1, W1, _ = img1_np.shape
        out_H = max(H0, H1)
        out_W = W0 + W1
        canvas = np.zeros((out_H, out_W, 3), dtype=img0_np.dtype)
        canvas[:H0, :W0, :] = img0_np
        canvas[:H1, W0:W0+W1, :] = img1_np

        fig = pl.figure(figsize=(10, 5)) # 調整大小以適應批次處理速度
        pl.imshow(canvas)

        # 準備線段
        p0 = matches_im0
        p1 = matches_im1.copy()
        p1[:, 0] += W0 # 右圖 X 座標偏移

        segments = np.stack([p0, p1], axis=1)
        
        # 顏色映射
        norm = pl.Normalize(MATCH_THRESHOLD, 1.0)
        colors = pl.cm.jet(norm(final_scores))

        lc = LineCollection(segments, colors=colors, linewidths=0.5, alpha=LINE_ALPHA)
        pl.gca().add_collection(lc)
        pl.axis('off')

        # 儲存
        pl.savefig(save_path, bbox_inches='tight', dpi=100)
        pl.close(fig) # 重要：關閉圖表釋放記憶體

    except Exception as e:
        print(f"[Vis Error] Failed to save matches for {save_path}: {e}")
        import traceback
        traceback.print_exc()

def process_single_pair(json_file, model):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            pose_i = data["poses"]["i"]
            pose_j = data["poses"]["j"]
            T_i = q_t_to_matrix(pose_i["q"], pose_i["t"])
            T_j = q_t_to_matrix(pose_j["q"], pose_j["t"])
            R_gt, t_gt = get_relative_pose(T_i, T_j)

            pil_i, pil_j = load_split_images(json_file)
            if pil_i is None:
                return None
            
            path_i = os.path.join(temp_dir, "img_i.jpg")
            path_j = os.path.join(temp_dir, "img_j.jpg")
            pil_i.save(path_i)
            pil_j.save(path_j)
            
            # 這裡 load_images 已經會自動轉 tensor 並 normalize
            imgs = load_images([path_i, path_j], size=512, verbose=False)
            pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)

            # === 資料轉換 ===
            view1_cpu, view2_cpu = pairs[0]

            def to_device_and_tensor(view_dict, device):
                new_view = {}
                for k, v in view_dict.items():
                    if k == 'img':
                        new_view[k] = v.to(device)
                    elif k == 'true_shape':
                        new_view[k] = torch.from_numpy(v).to(device)
                    else:
                        try:
                            if isinstance(v, np.ndarray):
                                new_view[k] = torch.from_numpy(v).to(device)
                            elif isinstance(v, torch.Tensor):
                                new_view[k] = v.to(device)
                            else:
                                new_view[k] = v
                        except:
                            new_view[k] = v
                return new_view

            view1_gpu = to_device_and_tensor(view1_cpu, DEVICE)
            view2_gpu = to_device_and_tensor(view2_cpu, DEVICE)

            # === Inference (Extract Features) ===
            with torch.no_grad():
                preds = model(view1_gpu, view2_gpu)

            # === Visualization (Matches) ===
            # 使用新的視覺化函數
            heatmap_filename = json_file.stem + "_matches.jpg" # 改個名字避免混淆
            heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
            
            save_matches_visualization(view1_gpu, view2_gpu, preds, heatmap_path)

            # === Pose Estimation ===
            optim_cache = os.path.join(temp_dir, "optim_cache")
            os.makedirs(optim_cache, exist_ok=True)

            scene = sparse_global_alignment(
                [path_i, path_j], 
                pairs,
                optim_cache,
                model,
                lr1=0.07, niter1=300,
                lr2=0.01, niter2=200,
                device=DEVICE,
                opt_depth=True,
                matching_conf_thr=0.0,
                verbose=False
            )

            poses_est = to_numpy(scene.get_im_poses())
            R_est, t_est = get_relative_pose(poses_est[0], poses_est[1])

            # Clean up
            del scene, pairs, imgs, preds, view1_gpu, view2_gpu
            torch.cuda.empty_cache()

            result_json = {
                "id": data.get("seq_id", "unknown") + "_" + json_file.stem,
                "heatmap_path": heatmap_path,
                "gt_pose": {
                    "R_ji_gt": R_gt.tolist(),
                    "t_ji_gt": t_gt.tolist()
                },
                "master_pose": {
                    "R_ji": R_est.tolist(),
                    "t_ji": t_est.tolist()
                }
            }
            return result_json

        except Exception as e:
            print(f"[Error] Failed on {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    
    print(f"Loading MASt3R model ({DEVICE})...")
    model = AsymmetricMASt3R.from_pretrained(MODEL_PATH).to(DEVICE)

    print(f"Scanning for JSON files in {PAIRS_ROOT}...")
    root_path = Path(PAIRS_ROOT)
    json_files = list(root_path.rglob("s*_i*_j*.json"))
    
    print(f"Found {len(json_files)} pairs.")

    for jf in tqdm(json_files, desc="Processing Pairs"):
        out_name = jf.name + "_pose_results.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # 檢查檔名對應新的 _matches.jpg
        heatmap_name = jf.stem + "_matches.jpg"
        heatmap_check_path = os.path.join(HEATMAP_DIR, heatmap_name)

        if os.path.exists(out_path) and os.path.exists(heatmap_check_path):
             continue

        result = process_single_pair(jf, model)

        if result:
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)

    print("All done!")

if __name__ == "__main__":
    main()