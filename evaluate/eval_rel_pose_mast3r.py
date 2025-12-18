"""
Relative Pose Estimation Evaluation Script

Computes AUC metrics for rotation and translation errors based on:
- Rotation error: Δθ_rot = arccos((tr(R_i^T @ R̂_i) - 1) / 2)
- Translation error: Δθ_trans = arccos((t̂_i^T @ t_i) / (||t̂_i|| * ||t_i||))
- Combined error: ε_i = max(Δθ_rot, Δθ_trans)
- AUC@ε_max computed via trapezoidal rule

Extra:
- Parse view angle bin from filename, e.g.
  "s00523_bin_135-180_i286_j509_rot139.0.json_pose_results.json"
  -> bin "135-180"
- Aggregate metrics per method & per view bin
- Plot AUC@5/10/20 vs view bin
- Export per-pair errors to CSV, sorted by max_error, with error-level bucket
"""

import json
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ===================== Basic error computations =====================

def compute_rotation_error(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """
    Compute rotation error in degrees.
    Δθ_rot = arccos((tr(R_gt^T @ R_est) - 1) / 2)
    """
    R_rel = R_gt.T @ R_est
    trace = np.trace(R_rel)
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_translation_error(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    """
    Compute translation error in degrees (angular error between directions).
    Δθ_trans = arccos((t̂^T @ t) / (||t̂|| * ||t||))
    """
    norm_gt = np.linalg.norm(t_gt)
    norm_est = np.linalg.norm(t_est)

    if norm_gt < 1e-8 or norm_est < 1e-8:
        return 0.0  # If either is zero, no meaningful direction error

    cos_angle = np.dot(t_est, t_gt) / (norm_est * norm_gt)
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_pose_error(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R_est: np.ndarray,
    t_est: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute rotation, translation, and combined pose errors.
    Returns: (rotation_error, translation_error, max_error) all in degrees
    """
    rot_err = compute_rotation_error(R_gt, R_est)
    trans_err = compute_translation_error(t_gt, t_est)
    max_err = max(rot_err, trans_err)
    return rot_err, trans_err, max_err


def compute_recall(errors: np.ndarray, threshold: float) -> float:
    """
    Compute recall at a given threshold.
    R(ε) = (1/N) * Σ 1{ε_i < ε}
    """
    return np.mean(errors < threshold)


def compute_auc(
    errors: np.ndarray,
    max_threshold: float,
    thresholds: Optional[List[float]] = None
) -> float:
    """
    Compute AUC using trapezoidal rule.
    AUC@ε_max ≈ (1/ε_max) * Σ [(R(ε_{j-1}) + R(ε_j)) / 2] * (ε_j - ε_{j-1})

    Uses fine-grained sampling (0.1 degree intervals) for accurate AUC computation.
    """
    if thresholds is None:
        thresholds = np.linspace(0, max_threshold, int(max_threshold * 10) + 1)
    else:
        thresholds = np.array(sorted(thresholds))
        if thresholds[0] != 0:
            thresholds = np.concatenate([[0], thresholds])

    recalls = [compute_recall(errors, t) for t in thresholds]
    auc = np.trapezoid(recalls, thresholds) / max_threshold
    return auc


# ===================== IO helpers =====================

def load_pose_results(json_path: str) -> Dict:
    """Load pose results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_view_bin_from_filename(path: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse view angle bin from filename.

    Example:
        "s00523_bin_135-180_i286_j509_rot139.0.json_pose_results.json"
        -> ("135-180", 135, 180)

    If not found, returns ("unknown", None, None).
    """
    name = Path(path).name
    m = re.search(r'_bin_(\d+)-(\d+)_', name)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        label = f"{start}-{end}"
        return label, start, end
    return "unknown", None, None


# ===================== Per-file evaluation =====================

def evaluate_single_file(data: Dict) -> Dict:
    """
    Evaluate pose estimation for a single file.
    Returns dict with errors for each method.
    """
    results = {'id': data.get('id', 'unknown')}

    # Ground truth relative pose (R_ji, t_ji)
    R_gt = np.array(data['gt_pose']['R_ji_gt'])
    t_gt = np.array(data['gt_pose']['t_ji_gt'])

    # Evaluate VGGT pose if available
    if 'vggt_pose' in data:
        R_vggt = np.array(data['vggt_pose']['R_ji'])
        t_vggt = np.array(data['vggt_pose']['t_ji'])
        rot_err, trans_err, max_err = compute_pose_error(R_gt, t_gt, R_vggt, t_vggt)
        results['vggt'] = {
            'rotation_error': rot_err,
            'translation_error': trans_err,
            'max_error': max_err
        }

    # Evaluate correspondence-based pose if available
    if 'corr_pose' in data:
        R_corr = np.array(data['corr_pose']['R_ji'])
        t_corr = np.array(data['corr_pose']['t_ji'])
        rot_err, trans_err, max_err = compute_pose_error(R_gt, t_gt, R_corr, t_corr)
        results['corr'] = {
            'rotation_error': rot_err,
            'translation_error': trans_err,
            'max_error': max_err,
            'num_inliers': data['corr_pose'].get('num_inliers', None)
        }

    return results


# ===================== Aggregation (overall + per-bin) =====================

def evaluate_all_files(json_files: List[str]) -> Dict:
    """
    Evaluate all pose result files and compute aggregate metrics.
    Also group by view angle bin (parsed from filename).
    """
    all_results = []
    view_bins_meta: Dict[str, Dict[str, Optional[int]]] = {}

    for json_path in json_files:
        try:
            data = load_pose_results(json_path)

            # Patch corr_pose if it contains None
            # if 'corr_pose' in data:
            #     corr_pose = data['corr_pose']
            #     if corr_pose.get('R_ji') is None:
            #         corr_pose['R_ji'] = np.eye(3, 3).tolist()
            #     if corr_pose.get('t_ji') is None:
            #         corr_pose['t_ji'] = np.zeros((3,)).tolist()
            #     data['corr_pose'] = corr_pose

            result = evaluate_single_file(data)

            # Attach filename & view bin info
            bin_label, bin_start, bin_end = parse_view_bin_from_filename(json_path)
            result['filename'] = json_path
            result['view_bin'] = bin_label

            if bin_label not in view_bins_meta:
                view_bins_meta[bin_label] = {
                    'start': bin_start,
                    'end': bin_end,
                }

            all_results.append(result)
        except Exception:
            # Skip problematic file silently
            continue

    if not all_results:
        return {}

    methods = ['vggt', 'corr']
    aggregate: Dict[str, Dict] = {}
    aggregate_by_bin: Dict[str, Dict[str, Dict]] = {m: {} for m in methods}

    for method in methods:
        rot_errors = []
        trans_errors = []
        max_errors = []

        # For per-bin accumulation
        bin_rot: Dict[str, List[float]] = {}
        bin_trans: Dict[str, List[float]] = {}
        bin_max: Dict[str, List[float]] = {}

        for result in all_results:
            if method in result:
                rot = result[method]['rotation_error']
                trans = result[method]['translation_error']
                mx = result[method]['max_error']

                rot_errors.append(rot)
                trans_errors.append(trans)
                max_errors.append(mx)

                bin_label = result.get('view_bin', 'unknown')
                bin_rot.setdefault(bin_label, []).append(rot)
                bin_trans.setdefault(bin_label, []).append(trans)
                bin_max.setdefault(bin_label, []).append(mx)

        if not max_errors:
            continue

        rot_errors_arr = np.array(rot_errors)
        trans_errors_arr = np.array(trans_errors)
        max_errors_arr = np.array(max_errors)

        # Overall metrics for this method
        aggregate[method] = {
            'num_samples': len(max_errors_arr),
            'rotation_error': {
                'mean': float(np.mean(rot_errors_arr)),
                'median': float(np.median(rot_errors_arr)),
                'std': float(np.std(rot_errors_arr))
            },
            'translation_error': {
                'mean': float(np.mean(trans_errors_arr)),
                'median': float(np.median(trans_errors_arr)),
                'std': float(np.std(trans_errors_arr))
            },
            'max_error': {
                'mean': float(np.mean(max_errors_arr)),
                'median': float(np.median(max_errors_arr)),
                'std': float(np.std(max_errors_arr))
            },
            'AUC@5': compute_auc(max_errors_arr, 5.0),
            'AUC@10': compute_auc(max_errors_arr, 10.0),
            'AUC@20': compute_auc(max_errors_arr, 20.0)
        }

        # Per-bin metrics
        aggregate_by_bin_method: Dict[str, Dict] = {}
        for bin_label, max_list in bin_max.items():
            max_arr = np.array(max_list)
            rot_arr = np.array(bin_rot[bin_label])
            trans_arr = np.array(bin_trans[bin_label])

            aggregate_by_bin_method[bin_label] = {
                'num_samples': len(max_arr),
                'rotation_error': {
                    'mean': float(np.mean(rot_arr)),
                    'median': float(np.median(rot_arr)),
                    'std': float(np.std(rot_arr))
                },
                'translation_error': {
                    'mean': float(np.mean(trans_arr)),
                    'median': float(np.median(trans_arr)),
                    'std': float(np.std(trans_arr))
                },
                'max_error': {
                    'mean': float(np.mean(max_arr)),
                    'median': float(np.median(max_arr)),
                    'std': float(np.std(max_arr))
                },
                'AUC@5': compute_auc(max_arr, 5.0),
                'AUC@10': compute_auc(max_arr, 10.0),
                'AUC@20': compute_auc(max_arr, 20.0)
            }

        aggregate_by_bin[method] = aggregate_by_bin_method

    return {
        'individual_results': all_results,
        'aggregate': aggregate,
        'aggregate_by_bin': aggregate_by_bin,
        'view_bins_meta': view_bins_meta
    }


# ===================== Pretty printing =====================

def print_results(results: Dict):
    """Pretty print evaluation results in table format."""
    if 'aggregate' not in results:
        print("No results to display.")
        return

    print("\n" + "=" * 80)
    print("RELATIVE POSE ESTIMATION EVALUATION RESULTS (OVERALL)")
    print("=" * 80)

    methods = list(results['aggregate'].keys())
    if not methods:
        print("No methods to display.")
        return

    header = f"{'Method':<15} | {'AUC@5°/10°/20°':<25} | {'Rot Err (mean)':<15} | {'Trans Err (mean)':<15}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    num_samples = None
    for method in methods:
        metrics = results['aggregate'][method]
        auc_str = f"{metrics['AUC@5']*100:.2f}/{metrics['AUC@10']*100:.2f}/{metrics['AUC@20']*100:.2f}"
        rot_err = f"{metrics['rotation_error']['mean']:.2f}"
        trans_err = f"{metrics['translation_error']['mean']:.2f}"
        num_samples = metrics['num_samples']
        print(f"{method.upper():<15} | {auc_str:<25} | {rot_err:<15} | {trans_err:<15}")

    print("-" * len(header))
    print(f"Number of samples (overall): {num_samples}")
    print("=" * 80)

    # Per-view-bin results
    if 'aggregate_by_bin' in results:
        print("\n" + "=" * 80)
        print("PER VIEW-ANGLE BIN RESULTS")
        print("=" * 80)

        aggregate_by_bin = results['aggregate_by_bin']
        view_bins_meta = results.get('view_bins_meta', {})

        def bin_sort_key(label: str) -> float:
            meta = view_bins_meta.get(label, {})
            start = meta.get('start')
            return float(start) if start is not None else 1e9

        all_bins = set()
        for method in aggregate_by_bin:
            all_bins.update(aggregate_by_bin[method].keys())
        sorted_bins = sorted(all_bins, key=bin_sort_key)

        for method in methods:
            if method not in aggregate_by_bin:
                continue

            print(f"\nMethod: {method.upper()}")
            header = f"{'View Bin':<15} | {'#Samples':<9} | {'AUC@5°':<8} | {'AUC@10°':<8} | {'AUC@20°':<8}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            for bin_label in sorted_bins:
                metrics = aggregate_by_bin[method].get(bin_label)
                if metrics is None:
                    continue
                num_samples = metrics['num_samples']
                auc5 = metrics['AUC@5'] * 100.0
                auc10 = metrics['AUC@10'] * 100.0
                auc20 = metrics['AUC@20'] * 100.0
                print(
                    f"{bin_label:<15} | "
                    f"{num_samples:<9d} | "
                    f"{auc5:>7.2f} | "
                    f"{auc10:>7.2f} | "
                    f"{auc20:>7.2f}"
                )

            print("-" * len(header))


# ===================== Plotting AUC vs view bin =====================

def plot_auc_by_view_bin(results: Dict, plot_dir: Path):
    """
    Plot AUC@5/10/20 vs view angle bin for each method.
    Saves PNG files to plot_dir.
    """
    if 'aggregate_by_bin' not in results or 'view_bins_meta' not in results:
        print("No per-bin results to plot.")
        return

    aggregate_by_bin = results['aggregate_by_bin']
    view_bins_meta = results['view_bins_meta']

    def bin_sort_key(label: str) -> float:
        meta = view_bins_meta.get(label, {})
        start = meta.get('start')
        return float(start) if start is not None else 1e9

    all_bins = set()
    for method in aggregate_by_bin:
        all_bins.update(aggregate_by_bin[method].keys())

    if not all_bins:
        print("No bins found for plotting.")
        return

    sorted_bins = sorted(all_bins, key=bin_sort_key)
    x = np.arange(len(sorted_bins))
    methods = list(aggregate_by_bin.keys())

    thresholds = [
        (5.0, 'AUC@5'),
        (10.0, 'AUC@10'),
        (20.0, 'AUC@20'),
    ]

    plot_dir.mkdir(parents=True, exist_ok=True)

    for max_thr, auc_key in thresholds:
        fig, ax = plt.subplots(figsize=(8, 4))

        width = 0.8 / max(len(methods), 1)

        for mi, method in enumerate(methods):
            auc_vals = []
            for bin_label in sorted_bins:
                metrics = aggregate_by_bin[method].get(bin_label)
                if metrics is None:
                    auc_vals.append(0.0)
                else:
                    auc_vals.append(metrics[auc_key])

            offsets = (mi - (len(methods) - 1) / 2) * width
            ax.bar(x + offsets, auc_vals, width, label=method.upper())

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_bins, rotation=45, ha='right')
        ax.set_ylabel('AUC')
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f'Relative Pose {auc_key} by View Angle Bin')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        fig.tight_layout()
        out_path = plot_dir / f'auc_by_viewbin_{int(max_thr)}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved plot: {out_path}")


# ===================== Per-pair CSV export for case study =====================

def error_bucket(max_error_deg: float) -> str:
    """
    Bucketize max_error into rough categories for case study.
    """
    if max_error_deg < 5.0:
        return "<5°"
    elif max_error_deg < 10.0:
        return "5°–10°"
    elif max_error_deg < 20.0:
        return "10°–20°"
    else:
        return ">20°"


def export_pair_errors(
    results: Dict,
    out_csv: Path,
    top_k: Optional[int] = None,
    methods: Optional[List[str]] = None
):
    """
    Flatten per-pair errors into a CSV, sorted by max_error (desc).

    Columns:
        filename, id, view_bin, method,
        rotation_error, translation_error, max_error, error_bucket
    """
    individual = results.get('individual_results', [])
    if not individual:
        print("No individual results, skip CSV export.")
        return

    if methods is None:
        methods = ['vggt', 'corr']

    rows = []
    for res in individual:
        filename = res.get('filename', '')
        pair_id = res.get('id', '')
        view_bin = res.get('view_bin', 'unknown')
        for m in methods:
            if m in res:
                e = res[m]
                max_err = float(e['max_error'])
                rows.append({
                    'filename': filename,
                    'id': pair_id,
                    'view_bin': view_bin,
                    'method': m,
                    'rotation_error': float(e['rotation_error']),
                    'translation_error': float(e['translation_error']),
                    'max_error': max_err,
                    'error_bucket': error_bucket(max_err),
                })

    # Sort by max_error desc (hard cases first)
    rows.sort(key=lambda r: r['max_error'], reverse=True)

    if top_k is not None and top_k > 0:
        rows = rows[:top_k]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'filename',
        'id',
        'view_bin',
        'method',
        'rotation_error',
        'translation_error',
        'max_error',
        'error_bucket',
    ]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-pair error CSV: {out_csv}")
    print(f"Total rows in CSV: {len(rows)}")


# ===================== CLI entry =====================

def main():
    parser = argparse.ArgumentParser(description='Evaluate relative pose estimation results')
    parser.add_argument(
        'folder',
        type=str,
        help='Directory containing JSON pose result files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file for aggregated results (optional)'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*_result.json',
        help='Glob pattern for finding JSON files in directory'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: <folder>/plots)'
    )
    parser.add_argument(
        '--pair-csv',
        type=str,
        default=None,
        help='Output CSV for per-pair errors (default: <folder>/pair_errors.csv)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Only keep top-K worst pairs in CSV (sorted by max_error). Default: all'
    )
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Provided path is not a directory: {folder_path}")
        return

    # Collect all JSON files inside the folder
    json_files = [str(f) for f in folder_path.glob(args.pattern)]

    if not json_files:
        print(f"No JSON files found in {folder_path} with pattern '{args.pattern}'")
        return

    print(f"Found {len(json_files)} pose result file(s) in {folder_path}")

    # Evaluate
    results = evaluate_all_files(json_files)

    # Print textual results
    print_results(results)

    # Save aggregated JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAggregated results saved to: {args.output}")

    # Plot per-bin AUC
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
    else:
        plot_dir = folder_path / "plots"
    plot_auc_by_view_bin(results, plot_dir)

    # Export per-pair CSV for case study
    if args.pair_csv is not None:
        csv_path = Path(args.pair_csv)
    else:
        csv_path = folder_path / "pair_errors.csv"

    export_pair_errors(results, csv_path, top_k=args.top_k)


if __name__ == '__main__':
    main()
