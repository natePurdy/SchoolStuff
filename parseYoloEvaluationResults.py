import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ────────────────────────────────────────────────
# CONFIG – change only this part
# ────────────────────────────────────────────────

TEXT_FILE_PATH = "/home/npurd/sandboxActual/sandbox/runs/segment/val/_yolov8l-seg.txt"   # ← paste your real path here
SAVE_FIG_DIR = Path("/home/npurd/sandboxActual/sandbox/runs/segment/val/val_metrics_plots")
SAVE_FIG_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────

def parse_aggregate_metrics(file_path):
    """Extract the 'all' line metrics from Ultralytics val output text"""
    box_p, box_r, box_map50, box_map = None, None, None, None
    mask_p, mask_r, mask_map50, mask_map = None, None, None, None

    pattern = re.compile(
        r"all\s+\d+\s+\d+\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Box
                box_p    = float(match.group(1))
                box_r    = float(match.group(2))
                box_map50 = float(match.group(3))
                box_map   = float(match.group(4))
                # Mask
                mask_p    = float(match.group(5))
                mask_r    = float(match.group(6))
                mask_map50 = float(match.group(7))
                mask_map   = float(match.group(8))
                break

    if box_map is None:
        raise ValueError("Could not find the 'all' metrics line in the file.")

    return {
        'box':   {'Precision': box_p,   'Recall': box_r,   'mAP@50': box_map50,   'mAP@50:95': box_map},
        'mask':  {'Precision': mask_p,  'Recall': mask_r,  'mAP@50': mask_map50,  'mAP@50:95': mask_map},
    }


def plot_metrics_bar(metrics_dict, title, filename):
    """Create simple bar plot for one set (box or mask)"""
    categories = ['Precision', 'Recall', 'mAP@50', 'mAP@50:95']
    values = [metrics_dict[c] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=['#4c78a8', '#f58518', '#e45756', '#72b7a1'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    save_path = SAVE_FIG_DIR / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print(f"Reading file: {TEXT_FILE_PATH}")

    try:
        metrics = parse_aggregate_metrics(TEXT_FILE_PATH)

        # Box plot
        plot_metrics_bar(
            metrics['box'],
            "Box Metrics - Aggregate (all classes)",
            "box_metrics_bar.png"
        )

        # Mask plot
        plot_metrics_bar(
            metrics['mask'],
            "Mask Metrics - Aggregate (all classes)",
            "mask_metrics_bar.png"
        )

        print("\nDone. Two figures created in:", SAVE_FIG_DIR.resolve())

    except Exception as e:
        print("Error:", e)
        print("Make sure the file contains a line like:")
        print("all 5000 36335 0.748 0.616 0.688 0.524 0.738 0.603 0.66 0.428")