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

INPUTS: COCO url,saving locations for data and log file
OUTPUT: downsized images and mask images, log file of how many "tiny" masks were destroyed during the downsizing to geta  feel for how much data was
"""

LOG_FILE = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/logImageDownsizing.txt"
with open(LOG_FILE, "w") as f:
    f.write("Image downsizing / segmentation loss log\n")
    f.write("=" * 60 + "\n\n")



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
    "horse",              # 18
    "sheep",              # 19
    "cow",                # 20
    "elephant",          # 21
    "bear",               # 22
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

def print_loss_stats(step, skipped_stats, dropped_instances_per_class, top_k=10):
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
            f.write(f"  {class_name:15s}: {cnt}\n")

        f.write("=" * 50 + "\n\n")

def process_single_image(args):
    image_id, annotations, base_url, output_image_dir, output_mask_dir, target_size = args

    local_skipped = Counter()
    local_dropped = Counter()
    local_kept = Counter()

    if not annotations:
        return False, local_skipped, local_dropped, local_kept

    filename = annotations[0]["file_name"]
    url = base_url + filename

    try:
        # --- Load image ---
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        W_orig, H_orig = img.size

        # --- Save resized image ---
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        img_resized.save(
            os.path.join(output_image_dir, filename.replace(".jpg", ".png"))
        )

        # --- Create full-res class mask ---
        mask_full = np.zeros((H_orig, W_orig), dtype=np.uint8)

        # Draw large objects first (matches your previous intent)
        annotations_sorted = sorted(
            annotations, key=lambda a: a.get("area", 0), reverse=True
        )

       

        for ann in annotations_sorted:
            cat_id = ann["category_id"]
            seg = ann.get("segmentation")

            if seg is None:
                local_skipped["no_segmentation"] += 1
                local_dropped[cat_id] += 1
                continue

            try:
                # ---- COCO-correct handling ----
                if isinstance(seg, list):
                    # Polygon(s)
                    rles = coco_mask.frPyObjects(seg, H_orig, W_orig)
                    rle = coco_mask.merge(rles)
                elif isinstance(seg, dict):
                    # RLE (iscrowd == 1)
                    rle = coco_mask.frPyObjects(seg, H_orig, W_orig)
                else:
                    raise ValueError("Unknown segmentation format")

                binary_mask = coco_mask.decode(rle)

                # Write class ID into mask
                mask_full[binary_mask == 1] = cat_id
                local_kept[cat_id] += 1

            except Exception as e:
                local_skipped["mask_decode_failed"] += 1
                local_dropped[cat_id] += 1

        # --- Resize mask ONCE using nearest neighbor ---
        mask_resized = Image.fromarray(mask_full).resize(
            (target_size, target_size),
            resample=Image.NEAREST
        )

        mask_np = np.array(mask_resized)

        # --- Optional: class survival stats ---
        orig_classes = {ann["category_id"] for ann in annotations if ann.get("segmentation")}
        resized_classes = set(np.unique(mask_np)) - {0}

        for cls in orig_classes - resized_classes:
            local_dropped[cls] += 1
            local_skipped["lost_after_resize"] += 1

        # --- Save mask ---
        mask_resized.save(
            os.path.join(output_mask_dir, filename.replace(".jpg", ".png"))
        )

        return True, local_skipped, local_dropped, local_kept

    except Exception:
        return False, Counter(), Counter(), Counter()


# target size is the number of rows and columns used to represent the downsized image.
def convertJsonsToBinariesAndSaveImagesAndMasks(pathToAnnotations,output_dir,base_url,target_size=256):

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
    num_workers = max(1, cpu_count() - 1)

    tasks = [
        (
            image_id,
            annotations,
            base_url,
            output_image_dir,
            output_mask_dir,
            target_size
        )
        for image_id, annotations in coco_dict.items()
    ]

    with Pool(num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for ok, s, d, k in pool.imap_unordered(process_single_image, tasks):
                # Update main counters
                skipped_stats.update(s)
                dropped_instances_per_class.update(d)
                kept_instances_per_class.update(k)

                count += 1
                pbar.update()

                # Write log every 100 images
                if count % 100 == 0:
                    print_loss_stats(count, skipped_stats, dropped_instances_per_class)    # ---- Save pickle ----
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
    print_loss_stats(count,skipped_stats,dropped_instances_per_class,top_k=15)

    return output_pickle




# ----------------- Save or Load Binary -----------------
json_train = f"/home/npurd/School/trainingData2/coco2017/2017coco/annotations/annotations/instances_train2017.json"
json_val = f"/home/npurd/School/trainingData2/coco2017/2017coco/annotations/annotations/instances_val2017.json"
image_base_url_TRAIN = f"http://images.cocodataset.org/train2017/"
image_base_url_VAL = f"http://images.cocodataset.org/val2017/"

# where to store downsized versions of all the images
imagesDownSizedTRAIN = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/train/"
imagesDownSizedVAL = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/val/"

# decide the size of the image you want to make them all
imageHW = 256

# cretes some nice binary files for sifting through and loading images using
# convert training and validation meta data to binary files (not images...)
binaryFileTrain = convertJsonsToBinariesAndSaveImagesAndMasks(json_train, imagesDownSizedTRAIN, image_base_url_TRAIN,  imageHW)
print(f"Converted {json_train} to binary...")
binaryFileVal = convertJsonsToBinariesAndSaveImagesAndMasks(json_val, imagesDownSizedVAL, image_base_url_VAL, imageHW)
print(f"Converted {json_val} to binary...")

