from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path
import re
import ultralytics
print("Ultralytics package location (which file is being imported):")
print("  ", ultralytics.__file__)
print("Ultralytics version:")
print("  ", ultralytics.__version__)
print("Ultralytics module path (should match above if single file):")
print("  ", ultralytics.__path__ if hasattr(ultralytics, '__path__') else "no __path__ (single-file module)")

# Full COCO class names (for reference only)
COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def extract_aggregate_metrics(metrics):
    """Extract key summary numbers – ultra-robust with full logging"""
    agg = {'box': {}, 'seg': {}}  # note: key is 'seg', not 'mask'

    print("\n=== DEBUG: Extracting aggregate metrics ===")

    for part_name in ['box', 'seg']:
        print(f"\n--- {part_name.upper()} ---")
        if not hasattr(metrics, part_name):
            print(f"  {part_name} DOES NOT EXIST")
            continue

        part = getattr(metrics, part_name)

        def safe_mean(attr_name):
            if not hasattr(part, attr_name):
                print(f"  {attr_name} not found")
                return None
            arr = getattr(part, attr_name)
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                print(f"  {attr_name} invalid or empty (type={type(arr)}, shape={getattr(arr, 'shape', 'N/A')})")
                return None
            try:
                arr_float = arr.astype(np.float64, copy=False)
                mean_val = np.nanmean(arr_float)
                print(f"  {attr_name} success: mean = {mean_val:.4f} (original shape={arr.shape})")
                return float(mean_val)
            except Exception as e:
                print(f"  {attr_name} FAILED mean: {e} (raw arr type={type(arr)}, first few={arr[:3] if arr.size > 0 else 'empty'})")
                return None

        def safe_get(attr_name):
            if not hasattr(part, attr_name):
                print(f"  {attr_name} not found")
                return None
            val = getattr(part, attr_name)
            try:
                if hasattr(val, 'item'):
                    val = val.item()
                float_val = float(val)
                print(f"  {attr_name} success: {float_val:.4f}")
                return float_val
            except Exception as e:
                print(f"  {attr_name} FAILED conversion: {e} (raw val={val}, type={type(val)})")
                return None

        agg[part_name] = {
            'Precision': safe_mean('p'),
            'Recall':    safe_mean('r'),
            'mAP@50':    safe_get('map50'),
            'mAP@50:95': safe_get('map'),
        }

    print("\nFinal extracted aggregates:")
    print("Box:", agg['box'])
    print("Seg (Mask):", agg['seg'])
    print("=====================================\n")

    return agg


def brutalDump(metrics, save_path: str):
    """Write aggregate metrics first, then full brutal dump"""
    agg_metrics = extract_aggregate_metrics(metrics)

    with open(save_path, 'w', encoding='utf-8') as f:
        # ── 1. Aggregate metrics summary ───────────────────────────────────────
        f.write("=== AGGREGATE METRICS SUMMARY ===\n\n")
        
        # Box
        f.write("Box:\n")
        box_dict = agg_metrics.get('box', {})
        if box_dict:
            for k, v in box_dict.items():
                val_str = f"{v:.4f}" if v is not None else "N/A"
                f.write(f"  {k:12} = {val_str}\n")
        else:
            f.write("  (No box metrics extracted)\n")

        # Mask (use 'seg' key!)
        f.write("\nMask:\n")
        mask_dict = agg_metrics.get('seg', {})  # FIXED: 'seg' not 'mask'
        if mask_dict:
            for k, v in mask_dict.items():
                val_str = f"{v:.4f}" if v is not None else "N/A"
                f.write(f"  {k:12} = {val_str}\n")
        else:
            f.write("  (No mask/seg metrics extracted)\n")
        
        f.write("\n" + "="*60 + "\n\n")

        # ── 2. Full brutal dump ────────────────────────────────────────────────
        old_options = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, linewidth=120, edgeitems=5, suppress=True)

        try:
            f.write("=== BRUTAL METRICS DUMP (FULL ARRAYS - NO TRUNCATION) ===\n\n")

            # Main metrics object
            for attr in sorted(dir(metrics)):
                if attr.startswith('_') or attr == 'confusion_matrix':
                    continue
                try:
                    value = getattr(metrics, attr)
                    if callable(value):
                        continue
                    f.write(f"{attr:24} = ")
                    if isinstance(value, (np.ndarray, list, tuple)):
                        if isinstance(value, np.ndarray):
                            f.write("\n" + str(value) + "\n")
                        else:
                            f.write(repr(value) + "\n")
                        if hasattr(value, '__len__'):
                            f.write(f"  (length={len(value)})")
                    else:
                        f.write(repr(value) + "\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"{attr:24} = <error: {e}>\n\n")

            # Box & Seg sub-objects
            for sub_name in ['box', 'seg']:
                if hasattr(metrics, sub_name):
                    sub = getattr(metrics, sub_name)
                    f.write(f"\n--- {sub_name.upper()} SUB-OBJECT ---\n")
                    for attr in sorted(dir(sub)):
                        if attr.startswith('_') or attr == 'confusion_matrix':
                            continue
                        try:
                            value = getattr(sub, attr)
                            if callable(value):
                                continue
                            f.write(f"  {attr:20} = ")
                            if isinstance(value, np.ndarray):
                                f.write("\n" + str(value) + "\n")
                            else:
                                f.write(repr(value) + "\n")
                            if hasattr(value, '__len__') and not isinstance(value, str):
                                f.write(f"  (length={len(value)})")
                            f.write("\n")
                        except:
                            pass
                    f.write("--- END ---\n\n")

            f.write("=== END BRUTAL DUMP ===\n\n")

        finally:
            np.set_printoptions(**old_options)

    print(f"Full file saved → {save_path}")


# ────────────────────────────────────────────────
# Main execution
# ────────────────────────────────────────────────

if __name__ == "__main__":
    modelDir = Path("/home/npurd/NN_MODELS")
    SAVE_DIR = Path("/home/npurd/sandboxActual/sandbox/runs/segment")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Saving to base dir: {SAVE_DIR.resolve()}\n")

    for modelname in os.listdir(modelDir):
        model_folder = Path(modelDir) / modelname
        if not model_folder.is_dir() or "yolo" not in modelname.lower():
            continue

        if "yolov5" in modelname.lower():
            print(f"\nSkipping YOLOv5 (incompatible with current Ultralytics): {modelname}")
            continue

        print(f"\n=== Version folder: {modelname} ===")

        # Define size order: smallest (n) = 0 → first, largest (x) = 4 → last
        size_order = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}

        def get_model_size_key(fn):
            fn_lower = fn.lower()
            # Pattern 1: yoloXXn-seg.pt, yolo11x.pt, etc.
            match = re.search(r'(?:yolo|v?\d+)[a-z]?([nsmlx])', fn_lower)
            if match:
                return size_order.get(match.group(1), 99)
            # Pattern 2: fallback - last n/s/m/l/x before -seg.pt or .pt
            match2 = re.search(r'([nsmlx])(?:-seg)?\.pt$', fn_lower)
            if match2:
                return size_order.get(match2.group(1), 99)
            return 99  # unknown last

        # Get and sort .pt files
        model_files = [
            f for f in os.listdir(model_folder) 
            if f.lower().endswith(".pt")
        ]

        if not model_files:
            print("  No .pt files found")
            continue

        sorted_models = sorted(model_files, key=get_model_size_key)

        print("  Processing order (smallest to largest):")
        for f in sorted_models:
            print(f"    - {f}")

        for model_filename in sorted_models:
            print(f"\n→ Processing: {modelname}/{model_filename}")
            
            MODEL_PATH = model_folder / model_filename
            YAML_PATH = "coco.yaml"

            model = YOLO(str(MODEL_PATH))
            print("  Running validation...")

            metrics = model.val(
                data=YAML_PATH,
                imgsz=640,
                batch=2,
                conf=0.001,
                iou=0.7,
                device=0,
                plots=True,
                save_json=True,
                # project="/home/npurd/sandboxActual/sandbox/runs/segment/test"
            )

            # Find latest val folder
            val_folders = [p for p in Path(SAVE_DIR).iterdir() 
                           if p.is_dir() and p.name.startswith("val")]
            if not val_folders:
                print("  Warning: No val* folder found")
                continue

            latest_folder = max(val_folders, key=lambda p: p.stat().st_mtime)
            print(f"  Latest val folder: {latest_folder.name}")

            save_path = latest_folder / f"_{model_filename.replace('.pt', '.txt')}"

            brutalDump(metrics, str(save_path))

            print(f"  Done → {save_path}\n")