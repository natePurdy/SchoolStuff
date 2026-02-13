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

    if not annotations: # skip images without annotations (there are about a thousand of them....)
        return False, local_skipped, local_dropped, local_kept

    filename = annotations[0]["file_name"]

    try:
        # --- Load image ---
        img = Image.open(os.path.join(baseInputFolder, filename)).convert("RGB")
        W_orig, H_orig = img.size

        # --- Save resized image ---
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        img_save_path = os.path.join(output_image_dir, filename.replace(".jpg", ".png"))
        img_resized.save(img_save_path, format="PNG", optimize=True, compress_level=4)

        # full resolution mask of classes for the image
        mask_full = np.zeros((H_orig, W_orig), dtype=np.uint8)

        # Draw large objects first so we preserve the small objects in front of large ones
        annotations_sorted = sorted(annotations, key=lambda a: a.get("area", 0), reverse=True)

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

        # collect original classes from image segmentation
        orig_classes = {ann["category_id"] for ann in annotations if ann.get("segmentation")}
        # look at what survived the resize mask by subtracting 
        resized_classes = set(np.unique(mask_np)) - {0} 

        # use the original set of labels to determine what has been lost in downsizing, and wont therefore be part of the images mask
        for cls in orig_classes - resized_classes:
            local_dropped[cls] += 1
            local_skipped["lost_after_resize"] += 1

        # --- Save mask ---
        mask_save_path = os.path.join(output_mask_dir, filename.replace(".jpg", ".png"))
        mask_resized.save(mask_save_path, format="PNG", optimize=True, compress_level=4)

        # make sure they didnt get corrupted
        if not verify_image_was_saved_correctly(img_save_path):
            print(f"!!! CORRUPTED OUTPUT IMAGE: {img_save_path}")
            # Optional: delete it so it doesn't pollute training
            try:
                os.remove(img_save_path)
            except:
                pass

        if not verify_image_was_saved_correctly(mask_save_path):
            print(f"!!! CORRUPTED OUTPUT MASK: {mask_save_path}")
            try:
                os.remove(mask_save_path)
            except:
                pass

        success = True

        # After both saves + verifications
        if not verify_image_was_saved_correctly(img_save_path):
            success = False
        if not verify_image_was_saved_correctly(mask_save_path):
            success = False


        return True, local_skipped, local_dropped, local_kept

    except Exception:
        return False, Counter(), Counter(), Counter()


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
                    print_loss_stats(count, skipped_stats, dropped_instances_per_class, kept_instances_per_class)    # ---- Save pickle ----
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
    print_loss_stats(count,skipped_stats,dropped_instances_per_class, kept_instances_per_class,top_k=15)

    return 

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
convertJsonsToBinariesAndSaveImagesAndMasks(json_train, imagesDownSizedTRAIN, image_base_dir_TRAIN,  imageHW)
print(f"Converted {json_train} to binary...")
convertJsonsToBinariesAndSaveImagesAndMasks(json_val, imagesDownSizedVAL, image_base_dir_VAL, imageHW)
print(f"Converted {json_val} to binary...")