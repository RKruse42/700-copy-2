#!/usr/bin/env python3
"""
Dataset Statistics Script for Minesweeper .npz datasets.

Calculates and reports:
  1. Calibration: Actual mine rate across 6 csp_prob buckets (p=0, 4 middle buckets, p=1).
  2. Distribution: Number and percentage of items in each bucket.
  3. Volume: Sample counts, unique seeds, and average number of samples per seed.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np


def compute_statistics(
    data_or_path: Union[str, Dict[str, np.ndarray], Any],
    closed: str = "left",
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    Computes calibration, distribution, and volume statistics from a dataset.

    Args:
        data_or_path: Path to .npz file or dictionary-like object with arrays.
        closed: Boundary convention for middle buckets ('left' for [a, b), 'right' for (a, b]).
        eps: Float tolerance for detecting exact 0.0 and 1.0 probability values.

    Returns:
        Dictionary containing volume statistics, bucket statistics, and summary metrics.
    """
    if isinstance(data_or_path, str):
        if not os.path.isfile(data_or_path):
            raise FileNotFoundError(f"Dataset file not found: {data_or_path}")
        data = np.load(data_or_path)
    else:
        data = data_or_path

    # Verify required keys
    required_keys = ["seeds", "csp_probs", "is_mine"]
    for key in required_keys:
        if key not in data:
            available = list(data.keys()) if hasattr(data, "keys") else data.files
            raise KeyError(
                f"Missing required key '{key}' in dataset. Available keys: {available}"
            )

    seeds = np.asarray(data["seeds"]).ravel()
    csp_probs = np.asarray(data["csp_probs"], dtype=np.float64).ravel()
    is_mine = np.asarray(data["is_mine"]).ravel().astype(bool)

    total_samples = len(csp_probs)
    if total_samples == 0:
        raise ValueError("Dataset is empty (0 samples).")

    if len(seeds) != total_samples or len(is_mine) != total_samples:
        raise ValueError(
            f"Array length mismatch: seeds={len(seeds)}, "
            f"csp_probs={len(csp_probs)}, is_mine={len(is_mine)}"
        )

    # 1. Volume Statistics
    unique_seeds, seed_counts = np.unique(seeds, return_counts=True)
    num_unique_seeds = len(unique_seeds)
    avg_samples_per_seed = (
        total_samples / num_unique_seeds if num_unique_seeds > 0 else 0.0
    )

    volume_stats = {
        "total_samples": total_samples,
        "num_unique_seeds": num_unique_seeds,
        "avg_samples_per_seed": avg_samples_per_seed,
        "min_samples_per_seed": int(np.min(seed_counts)) if num_unique_seeds > 0 else 0,
        "max_samples_per_seed": int(np.max(seed_counts)) if num_unique_seeds > 0 else 0,
        "median_samples_per_seed": float(np.median(seed_counts)) if num_unique_seeds > 0 else 0.0,
        "std_samples_per_seed": float(np.std(seed_counts)) if num_unique_seeds > 0 else 0.0,
    }

    # 2. Bucket Definitions (6 buckets total)
    # Bucket 0: p == 0.0 (safe)
    # Bucket 1..4: 4 buckets between 0.0 and 1.0 (step of 0.25)
    # Bucket 5: p == 1.0 (mine)
    is_zero = np.isclose(csp_probs, 0.0, atol=eps) | (csp_probs <= 0.0)
    is_one = np.isclose(csp_probs, 1.0, atol=eps) | (csp_probs >= 1.0)
    is_mid = (~is_zero) & (~is_one) & (csp_probs > 0.0) & (csp_probs < 1.0)

    # Boundaries for the 4 intermediate buckets: [0.0, 0.25, 0.50, 0.75, 1.0]
    buckets_def: List[Dict[str, Any]] = []

    # Bucket 0
    buckets_def.append({
        "id": 0,
        "label": "Bucket 0 (Safe)",
        "range_str": "p = 0.0",
        "mask": is_zero,
        "expected_mine_rate": 0.0,
    })

    # Middle 4 buckets
    if closed == "left":
        # [0.00, 0.25) -> (0.00, 0.25) because 0 is separate
        buckets_def.append({
            "id": 1,
            "label": "Bucket 1",
            "range_str": "0.00 < p < 0.25",
            "mask": is_mid & (csp_probs < 0.25),
            "expected_mine_rate": None,
        })
        # [0.25, 0.50)
        buckets_def.append({
            "id": 2,
            "label": "Bucket 2",
            "range_str": "0.25 <= p < 0.50",
            "mask": is_mid & (csp_probs >= 0.25) & (csp_probs < 0.50),
            "expected_mine_rate": None,
        })
        # [0.50, 0.75)
        buckets_def.append({
            "id": 3,
            "label": "Bucket 3",
            "range_str": "0.50 <= p < 0.75",
            "mask": is_mid & (csp_probs >= 0.50) & (csp_probs < 0.75),
            "expected_mine_rate": None,
        })
        # [0.75, 1.00)
        buckets_def.append({
            "id": 4,
            "label": "Bucket 4",
            "range_str": "0.75 <= p < 1.00",
            "mask": is_mid & (csp_probs >= 0.75),
            "expected_mine_rate": None,
        })
    else:  # right-closed
        # (0.00, 0.25]
        buckets_def.append({
            "id": 1,
            "label": "Bucket 1",
            "range_str": "0.00 < p <= 0.25",
            "mask": is_mid & (csp_probs <= 0.25),
            "expected_mine_rate": None,
        })
        # (0.25, 0.50]
        buckets_def.append({
            "id": 2,
            "label": "Bucket 2",
            "range_str": "0.25 < p <= 0.50",
            "mask": is_mid & (csp_probs > 0.25) & (csp_probs <= 0.50),
            "expected_mine_rate": None,
        })
        # (0.50, 0.75]
        buckets_def.append({
            "id": 3,
            "label": "Bucket 3",
            "range_str": "0.50 < p <= 0.75",
            "mask": is_mid & (csp_probs > 0.50) & (csp_probs <= 0.75),
            "expected_mine_rate": None,
        })
        # (0.75, 1.00) -> 1 is separate
        buckets_def.append({
            "id": 4,
            "label": "Bucket 4",
            "range_str": "0.75 < p < 1.00",
            "mask": is_mid & (csp_probs > 0.75),
            "expected_mine_rate": None,
        })

    # Bucket 5
    buckets_def.append({
        "id": 5,
        "label": "Bucket 5 (Mine)",
        "range_str": "p = 1.0",
        "mask": is_one,
        "expected_mine_rate": 1.0,
    })

    # Compute stats for each bucket
    bucket_stats: List[Dict[str, Any]] = []
    for b in buckets_def:
        mask = b["mask"]
        count = int(np.sum(mask))
        pct = (count / total_samples) * 100.0 if total_samples > 0 else 0.0

        if count > 0:
            actual_mines = int(np.sum(is_mine[mask]))
            mine_rate = actual_mines / count
            mean_prob = float(np.mean(csp_probs[mask]))
            calibration_gap = abs(mine_rate - mean_prob)
        else:
            actual_mines = 0
            mine_rate = None
            mean_prob = None
            calibration_gap = None

        bucket_stats.append({
            "id": b["id"],
            "label": b["label"],
            "range_str": b["range_str"],
            "count": count,
            "percentage": pct,
            "actual_mines": actual_mines,
            "mine_rate": mine_rate,
            "mean_prob": mean_prob,
            "calibration_gap": calibration_gap,
        })

    return {
        "volume": volume_stats,
        "buckets": bucket_stats,
        "closed": closed,
    }


def render_ascii_calibration_plot(bucket_stats: List[Dict[str, Any]], width: int = 30) -> str:
    """
    Renders an ASCII visualization showing how closely empirical mine rates
    track the predicted probabilities along the diagonal.
    """
    lines = []
    lines.append("Calibration Diagonal Tracking (Actual vs Expected):")
    lines.append("-" * 74)
    lines.append(f"{'Bucket':<18} | {'Expected (Prob)':<15} | {'Actual (Mine Rate)':<18} | {'Tracking':<15}")
    lines.append("-" * 74)

    for b in bucket_stats:
        if b["count"] == 0:
            lines.append(f"{b['label']:<18} | {'N/A (0 items)':<15} | {'N/A':<18} | [No Data]")
            continue

        exp_pct = b["mean_prob"] * 100.0
        act_pct = b["mine_rate"] * 100.0
        diff = act_pct - exp_pct

        # ASCII bar visualization
        pos_exp = min(width - 1, max(0, int(round((exp_pct / 100.0) * (width - 1)))))
        pos_act = min(width - 1, max(0, int(round((act_pct / 100.0) * (width - 1)))))

        bar = [" "] * width
        bar[pos_exp] = "|"
        if pos_act == pos_exp:
            bar[pos_act] = "*"  # Perfect match on diagonal
        else:
            bar[pos_act] = "o"

        bar_str = "".join(bar)
        diag_status = f"{diff:+5.1f}%" if abs(diff) >= 0.05 else "Exact"

        lines.append(
            f"{b['label']:<18} | {exp_pct:6.2f}%         | {act_pct:6.2f}% ({b['actual_mines']:,}/{b['count']:,}) | {diag_status:<6} [{bar_str}]"
        )

    lines.append("-" * 74)
    lines.append("Legend: '|' = Expected Probability, 'o' = Actual Mine Rate, '*' = Perfect Diagonal Match")
    return "\n".join(lines)


def print_dataset_statistics(stats: Dict[str, Any], dataset_name: str = "Dataset") -> None:
    """
    Prints a formatted report of Volume, Distribution, and Calibration.
    """
    vol = stats["volume"]
    buckets = stats["buckets"]
    closed_str = stats.get("closed", "left")

    sep = "=" * 82
    subsep = "-" * 82

    print(sep)
    print(f" DATASET STATISTICS: {dataset_name}")
    print(sep)

    # 1. VOLUME SECTION
    print("\n[VOLUME]")
    print(subsep)
    print(f"  Total samples collected:           {vol['total_samples']:,}")
    print(f"  Total unique seeds:                {vol['num_unique_seeds']:,}")
    print(f"  Average number of samples per seed: {vol['avg_samples_per_seed']:.2f}")
    if vol["num_unique_seeds"] > 1:
        print(
            f"  Samples per seed (Min / Med / Max): {vol['min_samples_per_seed']} / "
            f"{vol['median_samples_per_seed']:.1f} / {vol['max_samples_per_seed']} (std: {vol['std_samples_per_seed']:.2f})"
        )

    # 2. DISTRIBUTION SECTION
    print("\n[DISTRIBUTION]")
    print(subsep)
    print(f"{'Bucket':<18} | {'Range':<18} | {'Count':>12} | {'Share (%)':>10}")
    print(subsep)
    for b in buckets:
        print(f"{b['label']:<18} | {b['range_str']:<18} | {b['count']:>12,} | {b['percentage']:>9.2f}%")
    print(subsep)
    print(f"{'Total':<18} | {'All ranges':<18} | {vol['total_samples']:>12,} | {100.00:>9.2f}%")

    # 3. CALIBRATION SECTION
    print("\n[CALIBRATION]")
    print(f"(Interval partition mode: {closed_str}-closed)")
    print(subsep)
    print(
        f"{'Bucket':<18} | {'Range':<18} | {'Mean Prob':>11} | {'Actual Mine Rate':>18} | {'Status / Diagonal':>18}"
    )
    print(subsep)

    for b in buckets:
        if b["count"] == 0:
            status = "N/A (0 items)"
            mean_prob_str = "N/A"
            mine_rate_str = "N/A"
        else:
            mean_prob_str = f"{b['mean_prob'] * 100.0:6.2f}%"
            mine_rate_str = f"{b['mine_rate'] * 100.0:6.2f}%"

            if b["id"] == 0:
                status = "PASS (0.0% safe)" if b["mine_rate"] == 0.0 else "FAIL"
            elif b["id"] == 5:
                status = "PASS (100.0% mine)" if b["mine_rate"] == 1.0 else "FAIL"
            else:
                gap = (b["mine_rate"] - b["mean_prob"]) * 100.0
                status = f"gap: {gap:+5.2f}%"

        print(
            f"{b['label']:<18} | {b['range_str']:<18} | {mean_prob_str:>11} | {mine_rate_str:>18} | {status:>18}"
        )
    print(subsep)

    # Verification checklist
    b0 = buckets[0]
    b5 = buckets[5]
    b0_ok = b0["count"] == 0 or b0["mine_rate"] == 0.0
    b5_ok = b5["count"] == 0 or b5["mine_rate"] == 1.0
    print("\nVerification Checklist:")
    print(
        f"  [1] p=0.0 Bucket:  "
        f"{('PASS (0.0% mine rate)' if b0_ok else 'FAIL: ' + str(b0['mine_rate']))} "
        f"({b0['count']:,} samples, {b0['actual_mines']:,} mines)"
    )
    print(
        f"  [2] p=1.0 Bucket:  "
        f"{('PASS (100.0% mine rate)' if b5_ok else 'FAIL: ' + str(b5['mine_rate']))} "
        f"({b5['count']:,} samples, {b5['actual_mines']:,} mines)"
    )
    print("  [3] Middle Buckets: Tracking diagonal (see calibration chart below)")

    # 4. ASCII CALIBRATION CHART
    print("\n" + render_ascii_calibration_plot(buckets))
    print(sep + "\n")


def save_calibration_plot(stats: Dict[str, Any], output_path: str, dataset_name: str = "Dataset") -> None:
    """
    Saves a graphical calibration and distribution chart using Matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed. Cannot generate plot figure.")
        return

    buckets = stats["buckets"]
    vol = stats["volume"]

    labels = [b["label"].replace("Bucket ", "B") for b in buckets]
    counts = [b["count"] for b in buckets]
    mean_probs = [b["mean_prob"] if b["mean_prob"] is not None else np.nan for b in buckets]
    mine_rates = [b["mine_rate"] if b["mine_rate"] is not None else np.nan for b in buckets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Calibration Curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Diagonal)", alpha=0.7)
    valid_points = [
        (mp, mr, b["count"])
        for mp, mr, b in zip(mean_probs, mine_rates, buckets)
        if not np.isnan(mp) and not np.isnan(mr)
    ]
    if valid_points:
        xs, ys, cs = zip(*valid_points)
        ax1.plot(xs, ys, "s-", color="#1f77b4", linewidth=2, markersize=8, label="Dataset Empirical")
        for x, y, c in zip(xs, ys, cs):
            ax1.annotate(
                f"{y * 100:.1f}%\n(n={c})",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel("Predicted Probability (csp_prob)", fontsize=11)
    ax1.set_ylabel("Actual Mine Rate", fontsize=11)
    ax1.set_title(f"Calibration Reliability Diagram\n({dataset_name})", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Subplot 2: Bucket Distribution
    colors = ["#2ca02c", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#d62728"]
    bars = ax2.bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.set_title(
        f"Bucket Sample Distribution\n(Total: {vol['total_samples']:,}, Avg/Seed: {vol['avg_samples_per_seed']:.1f})",
        fontsize=12,
    )
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    for bar, c in zip(bars, counts):
        height = bar.get_height()
        pct = (c / vol["total_samples"]) * 100.0 if vol["total_samples"] > 0 else 0
        ax2.annotate(
            f"{c:,}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Calibration plot saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Calibration, Distribution, and Volume statistics for a Minesweeper .npz dataset."
    )
    parser.add_argument(
        "dataset_file",
        nargs="?",
        default="cos700_dataset.npz",
        help="Path to the .npz dataset file (default: cos700_dataset.npz)",
    )
    parser.add_argument(
        "--closed",
        choices=["left", "right"],
        default="left",
        help="Boundary convention for middle buckets: 'left' for [a, b) (default), 'right' for (a, b]",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="calibration_analysis.png",
        default=None,
        metavar="OUTPUT_IMAGE",
        help="Generate and save a calibration and distribution plot (e.g. calibration.png)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_file):
        print(f"Error: Dataset file '{args.dataset_file}' does not exist.", file=sys.stderr)
        return 1

    try:
        stats = compute_statistics(args.dataset_file, closed=args.closed)
    except Exception as exc:
        print(f"Error processing dataset '{args.dataset_file}': {exc}", file=sys.stderr)
        return 1

    print_dataset_statistics(stats, dataset_name=args.dataset_file)

    if args.plot:
        save_calibration_plot(stats, args.plot, dataset_name=os.path.basename(args.dataset_file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
