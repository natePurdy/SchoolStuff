import pickle
from collections import defaultdict
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Patch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from collections import Counter
from multiprocessing import Pool, cpu_count
from pycocotools import mask as coco_mask


# Keep track of what is being losst due to the downsizing of the images (masks basically dropping because they are too small to make sense really)
# really this is to get statistic about far awy object being dropped from the mask set i am creating due to downsampling- 
skipped_stats = Counter()
dropped_instances_per_class = Counter()
kept_instances_per_class = Counter()

"""
Okay so the point of this script is to save a simpler version of the coco sata set locally, so my pc can train at a reasonable rate for deeper nets
- NOTE: the catch is that the images have masks that are defined by polygon placement, and all the images are different sizes
- downsizeing images is okay, but downsizeing the masks is cuasing issues. 
i would like to save the masks as local copies (full sized to image), with the class labels as the integers placed in the mask file, so the
classes can be extracted from the masks, and locality relative to the color image will be true.
PROBLEM is that polygons will dissapear if they are being downsized to value lass than 3 (think about a polygons shape in terms of pixels.... it has a minimum)
so the mask image files after being downsized contain some bullshit noisy outlines of things, which will mess up the classes (they should just be solid object masks)
PROBLEM 2: the smaller images will dissapear completely, so far away objects will basically be removed from the training set unless they are somehow preserved (not sure if possible) - so
at the worst case the classifier will just suck at far away things in "the field", but should perform okay for training and is still 
probably a valid experiment if all the validation and training data is "corrupted" in this way

INPUTS: COCO image path,saving locations for data and log file
OUTPUT: downsized images and mask images, log file of how many "tiny" masks were destroyed during the downsizing to geta  feel for how much data was


NOTE: FIRST RUN: 
(navigate to where you want this data) then,
wget http://images.cocodataset.org/zips/train2017.zip    (and same for val)
wget http://images.cocodataset.org/zips/val2017.zip
(and also if you want annotations json...)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip

The
"""




# sorry...
COCO_CLASSES = [
    "background",        # 0
    "person",            # 1
    "bicycle",            # 2
    "car",                # 3
    "motorcycle",        # 4
    "airplane",          # 5
    "bus",                # 6
    "train",              # 7
    "truck",              # 8
    "boat",               # 9
    "traffic light",     # 10
    "fire hydrant",      # 11
    "stop sign",          # 12
    "parking meter",     # 13
    "bench",              # 14
    "bird",               # 15
    "cat",                # 16
    "dog",                # 17
    "horse",               # 18
    "sheep",              # 19
    "cow",                # 20
    "elephant",          # 21
    "bear",                # 22
    "zebra",              # 23
    "giraffe",           # 24
    "backpack",          # 25
    "umbrella",          # 26
    "handbag",           # 27
    "tie",                # 28
    "suitcase",          # 29
    "frisbee",           # 30
    "skis",               # 31
    "snowboard",         # 32
    "sports ball",       # 33
    "kite",               # 34
    "baseball bat",      # 35
    "baseball glove",    # 36
    "skateboard",        # 37
    "surfboard",         # 38
    "tennis racket",     # 39
    "bottle",             # 40
    "wine glass",        # 41
    "cup",                # 42
    "fork",               # 43
    "knife",              # 44
    "spoon",              # 45
    "bowl",               # 46
    "banana",             # 47
    "apple",              # 48
    "sandwich",          # 49
    "orange",             # 50
    "broccoli",          # 51
    "carrot",            # 52
    "hot dog",           # 53
    "pizza",             # 54
    "donut",             # 55
    "cake",               # 56
    "chair",              # 57
    "couch",              # 58
    "potted plant",      # 59
    "bed",                # 60
    "dining table",      # 61
    "toilet",            # 62
    "tv",                 # 63
    "laptop",            # 64
    "mouse",              # 65
    "remote",            # 66
    "keyboard",          # 67
    "cell phone",        # 68
    "microwave",         # 69
    "oven",               # 70
    "toaster",           # 71
    "sink",               # 72
    "refrigerator",      # 73
    "book",               # 74
    "clock",              # 75
    "vase",               # 76
    "scissors",          # 77
    "teddy bear",        # 78
    "hair drier",        # 79
    "toothbrush",        # 80
]

def print_loss_stats(step, skipped_stats, dropped_instances_per_class, kept_instances_per_class, top_k=10):
    with open(LOG_FILE, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"[After {step} images] Segmentation loss stats\n")

        total_skipped = sum(skipped_stats.values())
        f.write(f"Total skipped/lost segmentations: {total_skipped}\n")

        f.write("\nSkip reasons:\n")
        for k, v in skipped_stats.items():
            f.write(f"  {k:25s}: {v}\n")

        f.write("\nTop dropped classes:\n")
        for cls, cnt in dropped_instances_per_class.most_common(top_k):
            class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"ID {cls}"
            total_for_class = cnt + kept_instances_per_class.get(cls, 0)
            percent_lost = (cnt / total_for_class * 100) if total_for_class > 0 else 0
            f.write(f"  {class_name:15s}: {cnt} ({percent_lost:.2f}%)\n")

        f.write("=" * 50 + "\n\n")

def process_single_image(args):
    image_id, annotations, baseInputFolder, output_image_dir, output_mask_dir, target_size = args

    local_skipped = Counter()
    local_dropped = Counter()
    local_kept = Counter()

    if not annotations:
        return False, local_skipped, local_dropped, local_kept, {}, {}   # two empty dicts now

    filename = annotations[0]["file_name"]

    try:
        img = Image.open(os.path.join(baseInputFolder, filename)).convert("RGB")
        W_orig, H_orig = img.size

        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        img_save_path = os.path.join(output_image_dir, filename.replace(".jpg", ".png"))


        mask_full = np.zeros((H_orig, W_orig), dtype=np.uint8)

        annotations_sorted = sorted(annotations, key=lambda a: a.get("area", 0), reverse=True)

        for ann in annotations_sorted:
            cat_id = ann["category_id"]
            seg = ann.get("segmentation")

            if seg is None:
                local_skipped["no_segmentation"] += 1
                local_dropped[cat_id] += 1
                continue

            try:
                if isinstance(seg, list):
                    rles = coco_mask.frPyObjects(seg, H_orig, W_orig)
                    rle = coco_mask.merge(rles)
                elif isinstance(seg, dict):
                    rle = coco_mask.frPyObjects(seg, H_orig, W_orig)
                else:
                    raise ValueError("Unknown segmentation format")

                binary_mask = coco_mask.decode(rle)
                mask_full[binary_mask == 1] = cat_id
                local_kept[cat_id] += 1

            except Exception:
                local_skipped["mask_decode_failed"] += 1
                local_dropped[cat_id] += 1

        mask_resized = Image.fromarray(mask_full).resize(
            (target_size, target_size),
            resample=Image.NEAREST
        )

        mask_np = np.array(mask_resized)

        orig_classes = {ann["category_id"] for ann in annotations if ann.get("segmentation")}
        resized_classes = set(np.unique(mask_np)) - {0}

        for cls in orig_classes - resized_classes:
            local_dropped[cls] += 1
            local_skipped["lost_after_resize"] += 1

        # ────────────────────────────────────────────────────────────────
        # Pixel statistics in downsized mask
        # ────────────────────────────────────────────────────────────────
        unique, counts = np.unique(mask_np, return_counts=True)
        class_pixel_counts = dict(zip(unique, counts))

        total_foreground_pixels = sum(counts[unique != 0])   # exclude background

        avg_areas_this_image = {}     # pixels per instance
        avg_percent_this_image = {}   # % of foreground pixels per class (averaged per instance)

        for cls in resized_classes:
            if cls == 0:
                continue
            pixels = class_pixel_counts.get(cls, 0)
            orig_count = sum(1 for a in annotations if a["category_id"] == cls)

            if orig_count > 0 and pixels > 0:
                avg_pixels = pixels / orig_count
                avg_areas_this_image[cls] = avg_pixels

                # Percentage of foreground this class takes (per instance approximation)
                percent_per_instance = (pixels / total_foreground_pixels * 100) if total_foreground_pixels > 0 else 0
                avg_percent_this_image[cls] = percent_per_instance / orig_count   # avg % per instance

        # mask path
        mask_save_path = os.path.join(output_mask_dir, filename.replace(".jpg", ".png"))
        

        # verification code unchanged ...

        success = True
        if not verify_image_was_saved_correctly(img_save_path):
            success = False
        else:
            success = True # onlt save the mask and image if it is not corrupted after downsizing
            img_resized.save(img_save_path, format="PNG", optimize=True, compress_level=4)
            mask_resized.save(mask_save_path, format="PNG", optimize=True, compress_level=4)

        

        return success, local_skipped, local_dropped, local_kept, \
               avg_areas_this_image, avg_percent_this_image

    except Exception:
        return False, Counter(), Counter(), Counter(), {}, {}


# target size is the number of rows and columns used to represent the downsized image.
def convertJsonsToBinariesAndSaveImagesAndMasks(pathToAnnotations,output_dir,imageBaseDir,target_size=256):

    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # ---- Load JSON ----
    with open(pathToAnnotations, "r") as f:
        coco = json.load(f)

    # ---- Category mappings ----
    NAME_TO_TRAIN_ID = {cat["name"]: idx+1 for idx, cat in enumerate(coco["categories"])}
    COCO_ID_TO_TRAIN_ID = {cat["id"]: NAME_TO_TRAIN_ID[cat["name"]] for cat in coco["categories"]}
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # ---- Build annotations dict ----
    coco_dict = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        train_id = COCO_ID_TO_TRAIN_ID[ann["category_id"]]
        coco_dict[image_id].append({
            "segmentation": ann.get("segmentation"),
            "bbox": ann.get("bbox"),
            "category_id": train_id,
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd"),
            "annotation_id": ann.get("id"),
            "file_name": image_id_to_filename[image_id]
        })

    # ---- Download & save images and masks ----
    count = 0
    num_workers = max(1, cpu_count() - 1) # leave a thread to watch youtube while running

    tasks = [
        (
            image_id,
            annotations,
            imageBaseDir,
            output_image_dir,
            output_mask_dir,
            target_size
        )
        for image_id, annotations in coco_dict.items()
    ]

    class_avg_pixels_this_set = defaultdict(list)  
    class_avg_percent_this_set = defaultdict(list)

    with Pool(num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for ok, s, d, k, avg_pixels_dict, avg_percent_dict in pool.imap_unordered(process_single_image, tasks):
                skipped_stats.update(s)
                dropped_instances_per_class.update(d)
                kept_instances_per_class.update(k)

                for cls, val in avg_pixels_dict.items():
                    class_avg_pixels_this_set[cls].append(val)
                for cls, val in avg_percent_dict.items():
                    class_avg_percent_this_set[cls].append(val)

                count += 1
                pbar.update()

                if count % 100 == 0:
                    print_loss_stats(count, skipped_stats, dropped_instances_per_class, kept_instances_per_class)

    output_pickle = pathToAnnotations.replace(".json", ".pkl")
    to_save = {
        "annotations": dict(coco_dict),
        "category_id_to_name": {v: k for k, v in NAME_TO_TRAIN_ID.items()},
        "num_classes": 81
    }
    with open(output_pickle, "wb") as f:
        pickle.dump(to_save, f)
    print(f"Saved {len(coco_dict)} images with annotations to {output_pickle}")
    print(f"Downsized images saved to {output_image_dir}, masks saved to {output_mask_dir}")
    print_loss_stats(count, skipped_stats, dropped_instances_per_class, kept_instances_per_class, top_k=15)

    return class_avg_pixels_this_set, class_avg_percent_this_set

def verify_image_was_saved_correctly(filepath):
    """Returns True if file can be opened as valid image, False otherwise"""
    try:
        with Image.open(filepath) as im:
            im.verify()          # verifies internal PNG structure without loading pixels
            # Optional: force full decode
            _ = np.array(im)     # catches some decompression errors
        return True
    except Exception as e:
        print(f"Corruption detected in {filepath}: {type(e).__name__} - {str(e)}")
        return False


def plot_avg_instance_area_distribution(pixels_dict, percent_dict, dataset_name="Train", outputDir="", min_samples=1):
    """
    Side-by-side stem plots:
      Left: mean pixels per instance
      Right: mean % of foreground pixels per class (per instance)
    Shows all classes with ≥ min_samples
    """
    # ── Prepare data for pixels ──────────────────────────────────────
    valid_classes_p = []
    means_p = []
    stds_p = []
    counts_p = []

    for cls, areas in pixels_dict.items():
        if len(areas) >= min_samples:
            valid_classes_p.append(cls)
            means_p.append(np.mean(areas))
            stds_p.append(np.std(areas) if len(areas) > 1 else 0)
            counts_p.append(len(areas))

    # ── Prepare data for percentages ─────────────────────────────────
    valid_classes_pct = []
    means_pct = []
    stds_pct = []
    counts_pct = []

    for cls, percents in percent_dict.items():
        if len(percents) >= min_samples:
            valid_classes_pct.append(cls)
            means_pct.append(np.mean(percents))
            stds_pct.append(np.std(percents) if len(percents) > 1 else 0)
            counts_pct.append(len(percents))

    if not valid_classes_p and not valid_classes_pct:
        print(f"No data for {dataset_name}")
        return

    # Use union of classes (but usually same)
    all_classes = sorted(set(valid_classes_p) | set(valid_classes_pct))

    # For consistent ordering we'll sort by mean pixels descending
    if valid_classes_p:
        sort_key = {cls: mean for cls, mean in zip(valid_classes_p, means_p)}
    else:
        sort_key = {cls: 0 for cls in all_classes}

    sorted_classes = sorted(all_classes, key=lambda c: sort_key.get(c, 0), reverse=True)

    labels = [COCO_CLASSES[c] if c < len(COCO_CLASSES) else f"cls_{c}" for c in sorted_classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharex=True)

    # ── Left: absolute pixels ────────────────────────────────────────
    x = range(len(sorted_classes))
    y_pixels = [np.mean(pixels_dict.get(cls, [0])) for cls in sorted_classes]
    err_pixels = [np.std(pixels_dict.get(cls, [0])) for cls in sorted_classes]

    ax1.stem(x, y_pixels, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    ax1.errorbar(x, y_pixels, yerr=err_pixels, fmt="none", ecolor="gray", capsize=4, alpha=0.7)
    ax1.set_title("Mean pixels per instance")
    ax1.set_ylabel("Pixels")
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Right: percentage of foreground ──────────────────────────────
    y_pct = [np.mean(percent_dict.get(cls, [0])) for cls in sorted_classes]
    err_pct = [np.std(percent_dict.get(cls, [0])) for cls in sorted_classes]

    ax2.stem(x, y_pct, linefmt="C2-", markerfmt="C2o", basefmt=" ")
    ax2.errorbar(x, y_pct, yerr=err_pct, fmt="none", ecolor="gray", capsize=4, alpha=0.7)
    ax2.set_title("Mean % of foreground pixels per instance")
    ax2.set_ylabel("% of foreground")
    ax2.grid(True, axis="y", alpha=0.3)

    # Shared x-axis
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.tick_params(axis='x', which='major', labelsize=8)

    # Add n= counts on top of left plot (most informative)
    for i, cls in enumerate(sorted_classes):
        n = len(pixels_dict.get(cls, []))
        if n >= min_samples:
            y = y_pixels[i] + (err_pixels[i] or 0) + max(1, y_pixels[i]*0.02)
            ax1.text(i, y, f"n={n}", ha="center", va="bottom", fontsize=8, color="darkblue")

    fig.suptitle(f"Average Instance Size after Downsampling to {imageHW}x{imageHW} px - {dataset_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f"{outputDir}{dataset_name.lower()}_pixels_and_percent_stem_{imageHW}.png"
    plt.savefig(filename, dpi=140, bbox_inches="tight")
    print(f"Saved: {filename}")
    # plt.show()
    # plt.close()


def plot_combined_train_val_pixels_and_percent(train_pixels, train_percents,val_pixels, val_percents,min_samples=1,outputDir="",figsize=(24, 10)):
    """
    Creates ONE figure with FOUR stem subplots side-by-side:
      1. Train - Mean pixels per instance
      2. Val   - Mean pixels per instance
      3. Train - Mean % of foreground
      4. Val   - Mean % of foreground
    """
    # ── Helper to prepare data for one dataset ───────────────────────────────
    def prepare_data(pixels_dict, percent_dict):
        valid_classes = set(pixels_dict.keys()) | set(percent_dict.keys())
        data = []
        for cls in valid_classes:
            n = len(pixels_dict.get(cls, []))
            if n < min_samples:
                continue
            mean_pix = np.mean(pixels_dict.get(cls, [0]))
            std_pix  = np.std(pixels_dict.get(cls, [0])) if n > 1 else 0
            mean_pct = np.mean(percent_dict.get(cls, [0]))
            std_pct  = np.std(percent_dict.get(cls, [0])) if n > 1 else 0
            data.append((cls, mean_pix, std_pix, mean_pct, std_pct, n))
        return data

    train_data = prepare_data(train_pixels, train_percents)
    val_data   = prepare_data(val_pixels,   val_percents)

    if not train_data and not val_data:
        print("No data to plot")
        return

    # Get all classes that appear in at least one set with enough samples
    all_classes = sorted(set(d[0] for d in train_data + val_data))

    # Sort by mean pixels in train (descending) - or change to whatever you prefer
    sort_key = {d[0]: d[1] for d in train_data}  # mean pixels train
    all_classes.sort(key=lambda c: sort_key.get(c, 0), reverse=True)

    labels = [COCO_CLASSES[c] if c < len(COCO_CLASSES) else f"cls_{c}" for c in all_classes]

    # ── Create figure with 4 subplots ────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=False)
    fig.suptitle(f"Average Instance Size after Downsampling to {imageHW}x{imageHW} px\n"
                 f"Train vs Val comparison, classes with ≥ {min_samples} occurrences", fontsize=16)

    # Colors
    train_color = 'C0'   # blue-ish
    val_color   = 'C3'   # red-ish

    def plot_one(ax, data, color, title, plot_percent=False):
        if not data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        x = range(len(all_classes))

        if plot_percent:
            # Use mean_pct (index 3) and std_pct (index 4)
            y = [next((d[3] for d in data if d[0] == cls), 0) for cls in all_classes]
            err = [next((d[4] for d in data if d[0] == cls), 0) for cls in all_classes]
        else:
            # Use mean_pix (index 1) and std_pix (index 2)
            y = [next((d[1] for d in data if d[0] == cls), 0) for cls in all_classes]
            err = [next((d[2] for d in data if d[0] == cls), 0) for cls in all_classes]

        ax.stem(x, y, linefmt=f"{color}-", markerfmt=f"{color}o", basefmt=" ")
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, alpha=0.6, capsize=4)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)

        # Add n labels
        for i, cls in enumerate(all_classes):
            n = next((d[5] for d in data if d[0] == cls), 0)
            if n > 0:
                yval = y[i]
                errval = err[i]
                offset = max(1, yval * 0.03) if yval > 0 else 1
                ax.text(i, yval + errval + offset, f"n={n}", ha='center', va='bottom',
                        fontsize=8, color=color, alpha=0.9)

    # Plot the four subplots
    plot_one(axes[0], train_data, train_color, "Train  Mean pixels per instance")
    plot_one(axes[1], val_data,   val_color,   "Val    Mean pixels per instance")
    plot_one(axes[2], train_data, train_color, "Train  Mean % of foreground")
    plot_one(axes[3], val_data,   val_color,   "Val    Mean % of foreground")

    # Shared y-labels where possible
    axes[0].set_ylabel("Pixels per instance")
    axes[2].set_ylabel("% of foreground pixels")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"{outputDir}combined_train_val_pixels_percent_stem_{imageHW}.png"
    plt.savefig(filename, dpi=140, bbox_inches='tight')
    print(f"Saved combined plot: {filename}")
    plt.show()
    plt.close()


# ----------------- Save or Load Binary -----------------
json_train = f"/home/npurd/trainingData/COCO/annotations/instances_train2017.json"
json_val = f"/home/npurd/trainingData/COCO/annotations/instances_val2017.json"
image_base_dir_TRAIN = f"/home/npurd/trainingData/COCO/train2017"
image_base_dir_VAL = f"/home/npurd/trainingData/COCO/val2017"


# decide the size of the image you want to make them all
imageHW = 64 # square image them all
# where to store downsized versions of all the images
imagesDownSizedTRAIN = f"/home/npurd/trainingData/COCO/train2017_downsized{imageHW}/"
imagesDownSizedVAL = f"/home/npurd/trainingData/COCO/val2017_downsized{imageHW}/"




LOG_FILE = f"/home/npurd/trainingData/COCO/logImageDownsizing{imageHW}.txt"
with open(LOG_FILE, "w") as f:
    f.write("Image downsizing / segmentation loss log\n")
    f.write("=" * 60 + "\n\n")


# cretes some nice binary files for sifting through and loading images using
# convert training and validation meta data to binary files (not images...)
val_pixels, val_percents = convertJsonsToBinariesAndSaveImagesAndMasks(json_val, imagesDownSizedVAL, image_base_dir_VAL, imageHW)
plot_avg_instance_area_distribution(val_pixels, val_percents, dataset_name="Val", outputDir=imagesDownSizedVAL,  min_samples=1)

train_pixels, train_percents = convertJsonsToBinariesAndSaveImagesAndMasks(json_train, imagesDownSizedTRAIN, image_base_dir_TRAIN, imageHW)
plot_avg_instance_area_distribution(train_pixels, train_percents, dataset_name="Train", outputDir=imagesDownSizedTRAIN,  min_samples=1)


# broken  
# plot_combined_train_val_pixels_and_percent(
#     train_pixels, train_percents,
#     val_pixels, val_percents,
#     min_samples=1,
#     outputDir=imagesDownSizedTRAIN,
# )