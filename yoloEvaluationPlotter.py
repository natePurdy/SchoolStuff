from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import yaml # for the wierd yaml file with the class names mapped to indeces of data arrays...




def get_coco_class_names(yaml_path: Path = Path("/home/npurd/coco.yaml")):
    """Load class names from coco.yaml"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        names = data.get('names', {})
        if isinstance(names, dict):
            # {0: 'person', 1: 'bicycle', ...}
            return [names.get(i, f"class {i}") for i in range(80)]
        elif isinstance(names, list):
            return names + [""] * (80 - len(names))  # pad if shorter
        else:
            raise ValueError("Unexpected names format in yaml")
    except Exception as e:
        print(f"Warning: Could not load class names from {yaml_path}: {e}")
        return [f"Class {i}" for i in range(80)]

def extract_map_values(file_path: Path):
    """Extract MAP values by scanning lines and tracking current section (Box/Mask)."""
    try:
        lines = file_path.read_text(encoding='utf-8').splitlines()
        
        values = {}
        model_name = file_path.stem.lstrip('_').strip()  # e.g. "yolov8n-seg"
        
        current_section = None  # 'box' or 'mask'
        
        for line in lines[:60]:  # safe limit – top part only
            stripped = line.strip()
            if not stripped:
                continue
            
            # Detect section start
            if stripped.startswith("Box:"):
                current_section = 'box'
                continue
            if stripped.startswith("Mask:"):
                current_section = 'mask'
                continue
            
            # Skip if not in a known section
            if current_section is None:
                continue
            
            # Look for the keys (works even with extra spaces/tabs)
            if '=' in stripped:
                parts = [p.strip() for p in stripped.split('=', 1)]
                if len(parts) != 2:
                    continue
                key, val_str = parts
                
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                
                # Assign based on section and key
                if current_section == 'box':
                    if 'mAP@50' in key and ':' not in key:
                        values['box_map50'] = val
                    elif 'mAP@50:95' in key:
                        values['box_map'] = val
                elif current_section == 'mask':
                    if 'mAP@50' in key and ':' not in key:
                        values['mask_map50'] = val
                    elif 'mAP@50:95' in key:
                        values['mask_map'] = val
        
        # Require at least the two main mAP values
        if 'box_map' in values and 'mask_map' in values:
            values['model'] = model_name
            return values
        else:
            print(f"  → Missing key MAP values in {file_path.name} "
                  f"(found: {list(values.keys())})")
            return None
            
    except Exception as e:
        print(f"  Error reading {file_path.name}: {e}")
        return None

def extract_per_class_ap(file_path: Path):
    """Find and parse the all_ap array for box and mask.
       Returns per-class mAP@50:95 (average over the 10 IoU thresholds)."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Find box and mask all_ap blocks
        box_match = re.search(r'box\s*=\s*ultralytics\.utils\.metrics\.Metric object with attributes:.*?\n\s*all_ap:\s*array\((.*?)\)', 
                              content, re.DOTALL | re.IGNORECASE)
        mask_match = re.search(r'seg\s*=\s*ultralytics\.utils\.metrics\.Metric object with attributes:.*?\n\s*all_ap:\s*array\((.*?)\)', 
                               content, re.DOTALL | re.IGNORECASE)
        
        box_per_class = None
        mask_per_class = None
        
        if box_match:
            array_str = box_match.group(1).strip()
            array_str = array_str.replace('\n', '').replace('array(', '').replace(')', '').strip()
            try:
                full_ap = np.array(eval(array_str))          # shape should be (80, 10)
                if full_ap.ndim == 2 and full_ap.shape[1] == 10:
                    box_per_class = np.mean(full_ap, axis=1)   # ← THIS IS THE FIX
                    print(f"  [ok] Box per-class mAP extracted for {file_path.name} (mean={box_per_class.mean():.4f})")
                else:
                    print(f"  [error] Box array shape unexpected: {full_ap.shape}")
            except Exception as e:
                print(f"  Box AP parse failed in {file_path.name}: {e}")
        
        if mask_match:
            array_str = mask_match.group(1).strip()
            array_str = array_str.replace('\n', '').replace('array(', '').replace(')', '').strip()
            try:
                full_ap = np.array(eval(array_str))
                if full_ap.ndim == 2 and full_ap.shape[1] == 10:
                    mask_per_class = np.mean(full_ap, axis=1)
                    print(f"  [ok] Mask per-class mAP extracted for {file_path.name} (mean={mask_per_class.mean():.4f})")
                else:
                    print(f"  [error] Mask array shape unexpected: {full_ap.shape}")
            except Exception as e:
                print(f"  Mask AP parse failed in {file_path.name}: {e}")
        
        if box_per_class is not None or mask_per_class is not None:
            return {
                'model': file_path.stem.lstrip('_').strip(),
                'box_per_class': box_per_class,
                'mask_per_class': mask_per_class
            }
        return None
        
    except Exception as e:
        print(f"  Per-class AP error in {file_path.name}: {e}")
        return None

def plot_comparison(results: dict, metric: str):
    """
    Create a bar plot comparing the selected metric across all models.
    - Sorted from lowest to highest value
    - Bars colored by family: YOLOv8 (blues) vs YOLO11 (oranges)
    
    Args:
        results: dict of {model_name: metrics_dict}
        metric: one of 'box_map50', 'box_map', 'mask_map50', 'mask_map'
    """
    if metric not in ['box_map50', 'box_map', 'mask_map50', 'mask_map']:
        print(f"Invalid metric: {metric}. Choose one of: 'box_map50', 'box_map', 'mask_map50', 'mask_map'")
        return

    if not results:
        print("No data to plot.")
        return

    # Get list of (model, value) pairs and sort by value ascending
    model_value_pairs = [(m, results[m][metric]) for m in results]
    model_value_pairs.sort(key=lambda x: x[1])  # sort by value (lowest → highest)
    
    sorted_models = [pair[0] for pair in model_value_pairs]
    sorted_values = [pair[1] for pair in model_value_pairs]

    # Assign color based on model family (v8 vs v11)
    colors = []
    for m in sorted_models:
        if 'yolo11' in m.lower():
            colors.append('#ff7f0e')      # orange family for YOLO11
        else:
            colors.append('#1f77b4')      # blue family for YOLOv8

    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(sorted_models, sorted_values, color=colors, width=0.68)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.005,
            f'{height:.4f}', 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='bold'
        )

    ax.set_xlabel('Model (sorted by increasing performance)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} – Sorted Lowest to Highest')
    ax.set_xticks(range(len(sorted_models)))
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(sorted_values) * 1.15)  # headroom for labels

    # Optional: add a simple legend for the two families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='YOLOv8'),
        Patch(facecolor='#ff7f0e', label='YOLO11')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def plot_per_class_ap(per_class_results: dict, task: str = 'box'):
    """
    Plot per-class AP for the selected task ('box' or 'mask').
    - X-axis sorted by ascending average AP across models
    - Class names from coco.yaml
    - Models ordered: n < s < m < l < x   (smallest to largest)
    - Colors: lightest → darkest within each family
    """
    if not per_class_results:
        print("No per-class data available.")
        return

    # Load class names once
    class_names = get_coco_class_names()

    # Collect AP arrays
    model_aps = {}
    for model, data in per_class_results.items():
        ap_array = data.get(f'{task}_per_class')
        if ap_array is not None and len(ap_array) == 80:
            model_aps[model] = ap_array

    if not model_aps:
        print(f"No valid [{task}] per-class AP data.")
        return
    print("All detected model names:")
    for m in sorted(model_aps.keys()):
        print("   ", m)
    # Compute average AP per class (for sorting classes)
    avg_ap = np.mean(list(model_aps.values()), axis=0)  # shape (80,)
    
    # Sort classes by average performance
    sorted_indices = np.argsort(avg_ap)
    sorted_class_names = [class_names[i] for i in sorted_indices]

    # ── Hardcoded reliable order (n→s→m→l→x per family) ───────────────
    desired_order = [
        'yolov8n-seg',
        'yolov8s-seg',
        'yolov8m-seg',
        'yolov8l-seg',
        'yolov8x-seg',
        'yolo11n-seg',
        'yolo11s-seg',
        'yolo11m-seg',
        'yolo11l-seg',
        'yolo11x-seg',
    ]

    # Only keep models that actually exist in the data
    sorted_models = [m for m in desired_order if m in model_aps]

    # Split for color assignment
    sorted_v8 = [m for m in sorted_models if 'yolov8' in m]
    sorted_v11 = [m for m in sorted_models if 'yolo11' in m]

    print("Final plot order:", sorted_models)
    print("YOLOv8 in order:", sorted_v8)
    print("YOLO11 in order:", sorted_v11)

    # ── Colors: lightest to darkest within each family ────────────────
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_v8) or 1))
    oranges = plt.cm.Oranges(np.linspace(0.3, 0.9, len(sorted_v11) or 1))

    model_colors = {}
    for i, model in enumerate(sorted_v8):
        model_colors[model] = blues[i]
    for i, model in enumerate(sorted_v11):
        model_colors[model] = oranges[i]

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))

    for model in sorted_models:
        sorted_ap = model_aps[model][sorted_indices]
        color = model_colors.get(model, 'gray')
        is_v11 = 'yolo11' in model

        x_offset = 0.25 if is_v11 else 0.0
        x_axis = [i + x_offset for i in range(80)]

        ax.scatter(
            x_axis,
            sorted_ap,
            label=model,
            color=color,
            marker='.' if is_v11 else '.',
            s=28 if is_v11 else 24,
            alpha=0.92,
            linewidths=1.0
        )

    ax.set_xlabel('Classes (sorted by increasing average AP across models)')
    ax.set_ylabel('AP @50:95')
    ax.set_title(f'Per-Class mAP - {task.capitalize()} (sorted by avg performance)\nModels ordered: n < s < m < l < x')
    ax.set_xticks(range(80))
    ax.set_xticklabels(sorted_class_names, rotation=90, ha='right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def plot_modelSizesAveraged_per_class_ap(per_class_results: dict, task: str = 'box'):
    """
    Plot the AVERAGE per-class AP separately for YOLOv8 and YOLO11 model families.
    - X-axis sorted by ascending overall average AP across ALL models
    - Class names from coco.yaml
    - Two lines: one for YOLOv8 average (cold colors), one for YOLO11 average (hot colors)
    """
    if not per_class_results:
        print("No per-class data available.")
        return

    # Load class names
    class_names = get_coco_class_names()

    # Separate the models into two groups
    v8_aps = []
    v11_aps = []
    for model, data in per_class_results.items():
        ap_array = data.get(f'{task}_per_class')
        if ap_array is not None and len(ap_array) == 80:
            if 'yolo11' in model.lower():
                v11_aps.append(ap_array)
            else:
                v8_aps.append(ap_array)

    if not v8_aps and not v11_aps:
        print(f"No valid {task} per-class AP data from any model.")
        return

    # Compute averages per group
    avg_v8 = np.mean(v8_aps, axis=0) if v8_aps else None     # shape (80,)
    avg_v11 = np.mean(v11_aps, axis=0) if v11_aps else None

    # Compute overall average (used for sorting classes)
    all_aps = [a for a in [avg_v8, avg_v11] if a is not None]
    if not all_aps:
        print("No averages could be computed.")
        return

    overall_avg = np.mean(all_aps, axis=0)  # shape (80,)

    # Sort classes by ascending overall average AP
    sorted_indices = np.argsort(overall_avg)
    sorted_class_names = [class_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(16, 8))

    # ── YOLOv8 average (cold color) ───────────────────────────────────────────────
    if avg_v8 is not None:
        sorted_v8 = avg_v8[sorted_indices]
        ax.plot(range(80), sorted_v8, color='#1f77b4', linewidth=2.8,
                marker='o', markersize=6, alpha=0.95,
                label=f'YOLOv8 average')

    # ── YOLO11 average (hot color) ────────────────────────────────────────────────
    if avg_v11 is not None:
        sorted_v11 = avg_v11[sorted_indices]
        ax.plot(range(80), sorted_v11, color='#d62728', linewidth=2.8,
                marker='s', markersize=6, alpha=0.95,
                label=f'YOLO11 average')

    # Optional: add value labels every 5 classes (for the higher of the two lines)
    if avg_v8 is not None and avg_v11 is not None:
        higher_line = np.maximum(sorted_v8, sorted_v11)
        for i in range(0, 80, 5):
            val = higher_line[i]
            ax.text(i, val + 0.015, f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='black')

    ax.set_xlabel('Classes (sorted by increasing overall average AP)')
    ax.set_ylabel('Average AP @50:95')
    ax.set_title(f'Average Per-Class AP by Model Family, Averaged Over Model Family Sizes (n,s,m,l,x) {task.capitalize()}')
    ax.set_xticks(range(80))
    ax.set_xticklabels(sorted_class_names, rotation=90, ha='right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=11, bbox_to_anchor=(1.02, 1))
    ax.set_ylim(0, max(overall_avg) * 1.15 if overall_avg.size > 0 else 1.0)

    plt.tight_layout()
    plt.show()


def plot_delta_per_class_v8_vs_v11(per_class_results: dict, task: str = 'box'):
    """
    Plot per-class AP delta: YOLO11 - YOLOv8  (overall family average)
    Positive = YOLO11 better, negative = YOLOv8 better
    Classes sorted by average AP across all models (same as before)
    """
    if not per_class_results:
        print("No per-class data available.")
        return

    class_names = get_coco_class_names()

    v8_aps = []
    v11_aps = []
    for model, data in per_class_results.items():
        ap = data.get(f'{task}_per_class')
        if ap is not None and len(ap) == 80:
            if 'yolo11' in model.lower():
                v11_aps.append(ap)
            else:
                v8_aps.append(ap)

    if not v8_aps or not v11_aps:
        print(f"Cannot compute delta: missing data from one family ({task})")
        return

    avg_v8 = np.mean(v8_aps, axis=0)   # shape (80,)
    avg_v11 = np.mean(v11_aps, axis=0)

    # Delta: v11 - v8  → positive means v11 better
    delta = avg_v11 - avg_v8

    # Sort classes by overall average AP (same ordering as your main plot)
    overall_avg = (avg_v8 + avg_v11) / 2
    sorted_indices = np.argsort(overall_avg)
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_delta = delta[sorted_indices]

    fig, ax = plt.subplots(figsize=(16, 7))

    # Bar plot — red above zero, blue below
    colors = ['#d62728' if d >= 0 else '#1f77b4' for d in sorted_delta]

    bars = ax.bar(
        range(80),
        sorted_delta,
        color=colors,
        width=0.75,
        edgecolor='black',
        linewidth=0.4
    )

    # Zero line
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)

    # Add value labels on bars (only if |delta| > 0.005 to avoid clutter)
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.005:
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + (0.003 if height > 0 else -0.008),
                f'{height:+.3f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=8,
                fontweight='bold'
            )

    ax.set_xlabel('Classes (sorted by increasing average AP across all models)')
    ax.set_ylabel('Δ AP @50:95  (YOLO11 – YOLOv8)')
    ax.set_title(f'Per-Class AP Improvement: YOLO11 vs YOLOv8 Average  ({task.capitalize()})')
    ax.set_xticks(range(80))
    ax.set_xticklabels(sorted_class_names, rotation=90, ha='right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # Legend-like annotation
    ax.text(0.02, 0.98,
            "↑ YOLO11 better\n↓ YOLOv8 better",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_ylim(min(-0.05, sorted_delta.min() - 0.02), max(0.05, sorted_delta.max() + 0.02))

    plt.tight_layout()
    plt.show()


def plot_delta_per_class_by_size(per_class_results: dict, task: str = 'box'):
    """
    Plot per-class AP delta between comparable sizes: v11<size> - v8<size>
    One scatter series per size pair (n-n, s-s, m-m, l-l, x-x)
    Classes sorted by average AP across all models
    Legend and colors ordered from smallest to largest model size
    """
    if not per_class_results:
        print("No per-class data available.")
        return

    class_names = get_coco_class_names()

    # Explicit size suffix map - tailored to your "-seg" naming
    size_suffixes = ['n-seg', 's-seg', 'm-seg', 'l-seg', 'x-seg']

    size_to_v8 = {}
    size_to_v11 = {}

    for model, data in per_class_results.items():
        ap = data.get(f'{task}_per_class')
        if ap is None or len(ap) != 80:
            continue

        model_lower = model.lower()

        for sz_full in size_suffixes:
            sz = sz_full[0]  # 'n', 's', etc.
            if model_lower.endswith(sz_full):
                if 'yolo11' in model_lower:
                    size_to_v11[sz] = ap
                elif 'yolov8' in model_lower:
                    size_to_v8[sz] = ap
                break

    # Only keep sizes present in BOTH families
    common_sizes_set = set(size_to_v8.keys()) & set(size_to_v11.keys())

    if not common_sizes_set:
        print(f"No matching size pairs found for delta-by-size ({task})")
        print("Available v8 sizes:", sorted(size_to_v8.keys()))
        print("Available v11 sizes:", sorted(size_to_v11.keys()))
        return

    # Define desired size order: smallest to largest
    size_order = ['n', 's', 'm', 'l', 'x']

    # Sort the common sizes according to the desired order
    common_sizes = [sz for sz in size_order if sz in common_sizes_set]

    print(f"Computing size-paired deltas for sizes (ordered smallest → largest): {common_sizes}")

    # Compute deltas per size pair
    deltas_by_size = {}
    for sz in common_sizes:
        deltas_by_size[sz] = size_to_v11[sz] - size_to_v8[sz]   # shape (80,)

    # Use overall average for class sorting (same as your main plot)
    all_aps = [per_class_results[m][f'{task}_per_class'] for m in per_class_results]
    overall_avg = np.mean(all_aps, axis=0)
    sorted_indices = np.argsort(overall_avg)
    sorted_class_names = [class_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Colors from light to dark as size increases
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(common_sizes)))

    # Optional: add small horizontal offset so points don't perfectly overlap
    offset_step = 0.16

    for i, sz in enumerate(common_sizes):
        delta = deltas_by_size[sz][sorted_indices]
        
        # small jitter so series are easier to distinguish
        offset = (i - (len(common_sizes)-1)/2) * offset_step
        x_pos = np.arange(80) + offset

        ax.scatter(
            x_pos,
            delta,
            label=f'{sz.upper()}: v11 – v8',
            color=colors[i],
            marker='o',
            s=45,
            alpha=0.92,
            linewidths=0.8,
            edgecolor='black'
        )

        # Mean delta label on right side (aligned with offset)
        mean_d = np.mean(delta)
        ax.text(
            79.5 + offset, mean_d,
            f'mean {mean_d:+.3f}',
            ha='left', va='center',
            fontsize=9,
            color=colors[i],
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    ax.axhline(0, color='black', lw=1.4, ls='--', alpha=0.6)

    ax.set_xlabel('Classes (sorted by increasing average AP across all models)')
    ax.set_ylabel('Δ AP @50:95  (YOLO11 - YOLOv8)')
    ax.set_title(f'Per-Class AP Delta by Comparable Model Size  ({task.capitalize()})\nOrdered: nano → small → medium → large → xlarge')
    ax.set_xticks(range(80))
    ax.set_xticklabels(sorted_class_names, rotation=90, ha='right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)
    
    # Legend in upper left, outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title="Model Size")
    
    # Dynamic y-lim with padding
    all_deltas = np.concatenate(list(deltas_by_size.values()))
    ax.set_ylim(all_deltas.min() - 0.03, all_deltas.max() + 0.03)

    plt.tight_layout()
    plt.show()



def plot_average_delta_by_model_size(per_class_results: dict, task: str = 'box'):
    """
    Bar plot of average per-class AP delta (YOLO11 - YOLOv8) for each comparable model size.
    One bar per size (n, s, m, l, x) showing the mean Δ AP @50:95 across all 80 classes.
    Positive = YOLO11 better on average, Negative = YOLOv8 better on average.
    """
    if not per_class_results:
        print("No per-class data available.")
        return

    # Explicit size suffix map - matches your naming pattern
    size_suffixes = ['n-seg', 's-seg', 'm-seg', 'l-seg', 'x-seg']

    size_to_v8 = {}
    size_to_v11 = {}

    for model, data in per_class_results.items():
        ap = data.get(f'{task}_per_class')
        if ap is None or len(ap) != 80:
            continue

        model_lower = model.lower()

        for sz_full in size_suffixes:
            sz = sz_full[0]  # 'n', 's', etc.
            if model_lower.endswith(sz_full):
                if 'yolo11' in model_lower:
                    size_to_v11[sz] = ap
                elif 'yolov8' in model_lower:
                    size_to_v8[sz] = ap
                break

    # Only keep sizes present in BOTH families
    common_sizes_set = set(size_to_v8.keys()) & set(size_to_v11.keys())

    if not common_sizes_set:
        print(f"No matching size pairs found for average delta plot ({task})")
        print("Available v8 sizes:", sorted(size_to_v8.keys()))
        print("Available v11 sizes:", sorted(size_to_v11.keys()))
        return

    # Order sizes from smallest to largest
    size_order = ['n', 's', 'm', 'l', 'x']
    common_sizes = [sz for sz in size_order if sz in common_sizes_set]

    print(f"Plotting average deltas for sizes: {common_sizes}")

    # Compute mean delta for each size pair
    mean_deltas = []
    size_labels = []
    for sz in common_sizes:
        delta = size_to_v11[sz] - size_to_v8[sz]
        mean_deltas.append(np.mean(delta))          # average over 80 classes
        size_labels.append(f"{sz.upper()}")

    # Colors: light to dark as size increases
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(common_sizes)))

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        range(len(common_sizes)),
        mean_deltas,
        color=colors,
        width=0.65,
        edgecolor='black',
        linewidth=0.6
    )

    # Zero line
    ax.axhline(0, color='black', linewidth=1.1, linestyle='--', alpha=0.7)

    # Value labels on bars
    for bar, val in zip(bars, mean_deltas):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.002 if height >= 0 else -0.002
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + offset,
            f'{val:+.4f}',
            ha='center',
            va=va,
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xticks(range(len(common_sizes)))
    ax.set_xticklabels(size_labels, fontsize=11)

    ax.set_xlabel('Model Size')
    ax.set_ylabel(f'Mean Δ AP @50:95  (YOLO11 – YOLOv8)')
    ax.set_title(f'Average Per-Class AP Delta by Model Size\n({task.capitalize()}) – YOLO11 vs YOLOv8')
    
    # Light annotation
    ax.text(0.02, 0.98,
            "Positive = YOLO11 better on average\nNegative = YOLOv8 better on average",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    # Dynamic ylim with padding
    max_abs = max(abs(min(mean_deltas)), abs(max(mean_deltas)))
    ax.set_ylim(-max_abs * 1.25, max_abs * 1.25)

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def main():
    folder = Path("/home/npurd/sandboxActual/sandbox/runs/segment")
    plotHighLevelAP = False
    plotPerClassAP = True
    plotDeltas = True
    plotAveragesAmongSizes = False
    
    results = {}
    
    print("Scanning for .txt files in:", folder.resolve())
    txt_files = list(folder.rglob("*.txt"))
    if not txt_files:
        print("No .txt files found at all.")
        return
    
    for txt_file in sorted(txt_files):
        print(f"  Checking: {txt_file.name}")
        data = extract_map_values(txt_file)
        if data:
            results[data['model']] = data
    
    if not results:
        print("\nNo files had valid/extractable MAP values.")
        return
    
    # Print table (unchanged)
    print("\n" + "=" * 70)
    print("EXTRACTED MAP RESULTS")
    print("=" * 70)
    print(f"{'Model':<16} | Box mAP@50  | Box mAP    | Mask mAP@50  | Mask mAP")
    print("-" * 70)
    
    for model in sorted(results):
        d = results[model]
        print(f"{model:<16} | {d.get('box_map50', 'N/A'):>11.4f} | "
              f"{d.get('box_map', 'N/A'):>10.4f} | "
              f"{d.get('mask_map50', 'N/A'):>12.4f} | "
              f"{d.get('mask_map', 'N/A'):>10.4f}")
    
    print("=" * 70)

    # High-level plots
    print("\nGenerating plots...")
    if plotHighLevelAP:
        plot_comparison(results, 'box_map50')
        plot_comparison(results, 'box_map')
        plot_comparison(results, 'mask_map50')
        plot_comparison(results, 'mask_map')

    if plotPerClassAP:
        per_class_results = {}
        for txt_file in sorted(txt_files):
            per_class_data = extract_per_class_ap(txt_file)
            if per_class_data:
                per_class_results[per_class_data['model']] = per_class_data

        print("\nGenerating per-class AP plots...")
        plot_per_class_ap(per_class_results, task='box')
        plot_per_class_ap(per_class_results, task='mask')

        if plotDeltas == True:
            print("\nGenerating delta plots...")
            plot_delta_per_class_v8_vs_v11(per_class_results, task='box')
            plot_delta_per_class_v8_vs_v11(per_class_results, task='mask')
            plot_delta_per_class_by_size(per_class_results, task='box')
            plot_delta_per_class_by_size(per_class_results, task='mask')
            plot_average_delta_by_model_size(per_class_results, task='box')
            plot_average_delta_by_model_size(per_class_results, task='mask')

        if plotAveragesAmongSizes ==True:
            print("\nGenerating average per-class AP plots...")
            plot_modelSizesAveraged_per_class_ap(per_class_results, task='box')
            plot_modelSizesAveraged_per_class_ap(per_class_results, task='mask')
if __name__ == "__main__":
    main()