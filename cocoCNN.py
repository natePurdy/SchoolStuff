import pickle
from collections import defaultdict
from skimage.measure import label as sk_label
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import requests
from io import BytesIO
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Patch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import sys
from collections import defaultdict

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

BASE_COLORS = np.array([
    [230, 25, 75],    # red
    [60, 180, 75],    # green
    [255, 225, 25],   # yellow
    [0, 130, 200],    # blue
    [245, 130, 48],   # orange
    [145, 30, 180],   # purple
    [70, 240, 240],   # cyan
    [240, 50, 230],   # magenta
    [210, 245, 60],   # lime
    [250, 190, 212],  # pink
    [0, 128, 128],    # teal
    [220, 190, 255],  # lavender
    [170, 110, 40],   # brown
    [255, 250, 200],  # beige
    [128, 0, 0],      # maroon
    [170, 255, 195],  # mint
    [128, 128, 0],    # olive
    [255, 215, 180],  # apricot
    [0, 0, 128],      # navy
    [128, 128, 128],  # gray
], dtype=np.uint8)

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True     # important for conv layers etc.
    torch.backends.cudnn.benchmark = False

# Call this early
set_global_seed(12345)   # pick any fixed number
def custom_cutout(img, rng, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    if rng.random() > p:
        return img
    
    h, w = img.shape[1], img.shape[2]
    area = h * w
    
    for _ in range(100):  # num retries to find valid rect
        erase_area = area * rng.uniform(*scale)
        aspect_ratio = rng.uniform(*ratio)
        
        erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(erase_area / aspect_ratio)))
        
        if erase_h >= h or erase_w >= w:
            continue
        
        x = rng.randint(0, w - erase_w)
        y = rng.randint(0, h - erase_h)
        
        # random value per channel (like value="random")
        v = rng.uniform(0, 1, size=(3, 1, 1))
        
        img[:, y:y+erase_h, x:x+erase_w] = v
        break  # success
    
    return img

# for grading heatmap poverlay to truth data label overlay (not sure if there is a better way to judge)
def segmentation_loss(logits, target, num_classes=81, dice_weight=0.5, eps=1e-8):
    # Cross-entropy (ignores background via ignore_index=0)
    ce = nn.functional.cross_entropy(logits, target, ignore_index=0, reduction='mean')

    # Softmax over classes
    probs = torch.softmax(logits, dim=1)           # (B, C, H, W)

    # One-hot target (only foreground classes matter)
    target_onehot = torch.zeros_like(probs)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)   # (B, C, H, W)

    # Ignore background channel (index 0)
    probs_fg   = probs[:, 1:]                         # (B, 80, H, W)
    target_fg  = target_onehot[:, 1:]                 # (B, 80, H, W)

    # Flatten spatial dims
    probs_fg   = probs_fg.reshape(probs_fg.size(0), probs_fg.size(1), -1)   # (B, 80, H*W)
    target_fg  = target_fg.reshape(target_fg.size(0), target_fg.size(1), -1)

    # Intersection and union per class per image
    inter = (probs_fg * target_fg).sum(dim=2)         # (B, 80)
    union = probs_fg.sum(dim=2) + target_fg.sum(dim=2)

    # Dice coefficient per class per image
    dice = (2. * inter + eps) / (union + eps)         # (B, 80)
    dice_loss = 1 - dice                              # (B, 80)

    # Average only over classes that appear in GT (per image)
    valid = target_fg.sum(dim=2) > 0                  # (B, 80) — classes present
    dice_loss = dice_loss * valid.float()
    dice_loss = dice_loss.sum(dim=1) / (valid.sum(dim=1) + eps)   # mean per image

    dice_loss = dice_loss.mean()                      # mean over batch

    return ce + dice_weight * dice_loss

# for loading data and using the DataLoader torch class
class COCOSegmentationDataset(Dataset):
    def __init__(self,fileList,image_folder,mask_folder,target_size=256,augment=True,num_classes=81,rand_seed=42, doCutouts=False):
        self.fileList = fileList
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_size = target_size
        self.augment = augment
        self.num_classes = num_classes
        self.doCutouts = doCutouts
        
        # Reproducible randomness for augmentations
        self.rngForAugs = np.random.RandomState(rand_seed)
        self.randaug = T.RandAugment(num_ops=2, magnitude=9)  # magnitude 9-10 is quite strong

        # for possible failed images (should be any since my downsizing script checks for the issues that cause me to put this in here!!!)
        self.dummy_img  = torch.zeros(3, target_size, target_size, dtype=torch.float32)
        self.dummy_mask = torch.zeros(target_size, target_size, dtype=torch.long)

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        filename = self.fileList[idx]
        img_path = os.path.join(self.image_folder, filename)
        mask_path = os.path.join(self.mask_folder, filename)

        # try and do normal laoding routine (hopefully no images are corrpted but my experience says yes there willl be always)
        try:
            # Load as PIL
            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")   # class IDs as grayscale

            # Geometric augmentations ── applied to BOTH image and mask
            if self.augment:
                # 1. Horizontal flip (50% chance)
                if self.rngForAugs.random() > 0.5:
                    img  = img.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

                # 2. Random crop with padding (like CIFAR training)
                pad = int(self.target_size/16)  # larger pad than CIFAR because your images are 256x256
                img  = ImageOps.expand(img,  border=pad, fill=0)
                mask = ImageOps.expand(mask, border=pad, fill=0)  # fill=0 = background class

                # Random crop back to target_size
                left = self.rngForAugs.randint(0, 2 * pad)
                top  = self.rngForAugs.randint(0, 2 * pad)
                img  = img.crop((left, top, left + self.target_size, top + self.target_size))
                mask = mask.crop((left, top, left + self.target_size, top + self.target_size))

                # Optional: add small random rotation / shear if desired
                # angle = self.rng.uniform(-15, 15)
                # img  = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                # mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

                # Color / style augmentations  ONLY on image
    
                # RandAugment (very effective  applies random sequence of strong ops)
                img = self.randaug(img)
                factor = 0.8 + self.rngForAugs.random() * 0.4  # brightness
                img = ImageEnhance.Brightness(img).enhance(factor)
                factor = 0.8 + self.rngForAugs.random() * 0.4  # contrast
                img = ImageEnhance.Contrast(img).enhance(factor)
                factor = 0.8 + self.rngForAugs.random() * 0.4  # saturation
                img = ImageEnhance.Color(img).enhance(factor)


            # to numpy for cutout, then tensor after
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (C,H,W)
            if self.doCutouts == True:
                img = custom_cutout(img, self.rngForAugs, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            else:
                img=img
            img_tensor = torch.from_numpy(img)

            # convert from PIL to numpy to tensor
            mask_np = np.array(mask, dtype=np.int64)
            mask_tensor = torch.from_numpy(mask_np).long()
            return img_tensor, mask_tensor, filename
        except Exception as e:
            print(f"Skipping corrupted file: {filename} |  error: {type(e).__name__} - {str(e)}")
            # log file why not
            with open("corrupted_files.txt", "a") as f:
                f.write(f"{filename} | {type(e).__name__}: {str(e)}\n")

            # Return dummy sample so batch doesn't break
            return self.dummy_img.clone(), self.dummy_mask.clone(), f"SKIPPED_{filename}"


class ResidualBlock(nn.Module):
    """Simple pre-activation residual block (you can replace with your own version)"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity


class RCNNSegmentation(nn.Module):
    def __init__(
        self,
        num_classes=81,
        input_channels=3,           # renamed from channelDepth for clarity
        base_channels=32,
        num_stages=3,               # ← NEW: how many down/up stages (was fixed at 3)
        kernel_sizes=None,          # ← NEW: list of kernel sizes, one per stage
        residual_blocks_per_stage=2,
        input_height=64,
        input_width=64,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3] * num_stages   # default: 3×3 everywhere
        if len(kernel_sizes) != num_stages:
            raise ValueError(f"kernel_sizes ({len(kernel_sizes)}) must match num_stages ({num_stages})")

        # Check divisibility — now generalized for arbitrary num_stages
        divisor = 2 ** num_stages
        if input_height % divisor != 0 or input_width % divisor != 0:
            raise ValueError(
                f"Input size must be divisible by 2^{num_stages} = {divisor} "
                f"(got {input_height}x{input_width})"
            )

        self.expected_h = input_height
        self.expected_w = input_width

        # Build channel list: [32, 64, 128, 256, ...]
        channels = [base_channels * (2 ** i) for i in range(num_stages + 1)]
        # channels[0] = first conv out   channels[1] = after stage 1, etc.
        # channels[num_stages] = bottleneck

        # ────────────────────────────────────────────────
        #                   Encoder
        # ────────────────────────────────────────────────
        encoder_layers = []

        # First conv (from input channels → base_channels)
        encoder_layers.extend([
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        ])

        # Stages 1 to num_stages
        for i in range(num_stages):
            ch_in  = channels[i]
            ch_out = channels[i+1]
            k = kernel_sizes[i]

            # Residual blocks (same channels)
            for _ in range(residual_blocks_per_stage):
                encoder_layers.append(ResidualBlock(ch_in, kernel_size=k))

            # Transition to next channel count (1×1 conv when channels change)
            if ch_in != ch_out:
                encoder_layers.append(nn.Conv2d(ch_in, ch_out, 1, bias=False))
                encoder_layers.append(nn.BatchNorm2d(ch_out))
                encoder_layers.append(nn.ReLU(inplace=True))

            # Downsample (except after last stage)
            if i < num_stages - 1:
                encoder_layers.append(nn.MaxPool2d(2))

        self.encoder = nn.Sequential(*encoder_layers)

        # ────────────────────────────────────────────────
        #                   Decoder
        # ────────────────────────────────────────────────
        decoder_layers = []

        current_ch = channels[-1]   # bottleneck

        # Upsample only num_stages-1 times (match the number of actual downsamples)
        for i in reversed(range(num_stages - 1)):  # i = 1,0 for num_stages=3
            next_ch = channels[i]

            decoder_layers.extend([
                nn.ConvTranspose2d(current_ch, next_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(inplace=True),
            ])

            # Residual blocks at this resolution
            for _ in range(residual_blocks_per_stage):
                decoder_layers.append(ResidualBlock(next_ch, kernel_size=kernel_sizes[i]))

            current_ch = next_ch

        # Final 1×1 → classes (no more upsampling)
        decoder_layers.append(nn.Conv2d(current_ch, num_classes, kernel_size=1))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if x.shape[2] != self.expected_h or x.shape[3] != self.expected_w:
            raise RuntimeError(
                f"Expected ...x{self.expected_h}x{self.expected_w}, got {x.shape}"
            )
        return self.decoder(self.encoder(x))
#  Function to visualize a few images with masks (minor tweaks for torch tensors)
# Function to visualize a few images with masks
def show_images_with_masks(dataset, BASE_COLORS, n=5):

    # Normalize colors for matplotlib (0-1)
    BASE_COLORS = BASE_COLORS / 255.0


    
    for i in range(n):
        plt.figure(figsize=(12, 6))
        img, mask, fname = dataset[i]  
        img_np = img.numpy().transpose(1,2,0)  # (H,W,C)
        mask_np = mask.numpy()  # CHANGED: To numpy for viz

        # --- Get class names present in the mask ---
        unique_ids = np.unique(mask_np)
        unique_ids = unique_ids[unique_ids != 0]  # remove background
        class_names = [COCO_CLASSES[int(cls)] if cls < len(COCO_CLASSES) else f"ID {int(cls)}"
                       for cls in unique_ids]

        # --- Show image ---
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title(f"{fname} - Image")
        plt.axis('off')

        # --- Show mask with custom colors ---
        # Build RGB mask
        mask_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)
        for cls in unique_ids:
            color = BASE_COLORS[int(cls) % len(BASE_COLORS)]
            mask_rgb[mask_np == cls] = color

        plt.subplot(1,2,2)
        plt.imshow(mask_rgb)
        plt.title(f"{fname} - Mask")
        plt.axis('off')

        # --- Add legend ---
        patches = [Patch(color=BASE_COLORS[int(cls) % len(BASE_COLORS)], label=name)
                   for cls, name in zip(unique_ids, class_names)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.show()

# Function to compute mean IoU ( "grading" metric for mask intersection) --> vectorized and save massive time during training because its vectorized
def compute_miou(preds,labels,current_class_mious: dict[int, float],  num_classes: int = 81,ignore_index: int = 0) -> float:
    """
    Compute mean IoU over foreground classes present in this batch.
    Also updates current_class_mious with the per-class IoU from *this batch*
    for every class that appears in either prediction or ground truth.

    Returns: mean IoU of classes that appeared in this batch (or 0.0 if none)
    """
    batch_class_ious = {}  # class → iou for this batch only

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_mask = (preds == c)
        gt_mask   = (labels == c)

        intersection = (pred_mask & gt_mask).sum().float()
        union        = (pred_mask | gt_mask).sum().float()

        if union > 0:
            iou = intersection / union
            iou_value = iou.item()
            batch_class_ious[c] = iou_value

    if not batch_class_ious:
        return 0.0

    # Update the persistent dictionary with this batch's values
    current_class_mious.update(batch_class_ious)

    # Return mean over classes that appeared *in this batch*
    return np.mean(list(batch_class_ious.values()))

def compute_ap_per_class(preds, labels, probs, num_classes=81):
    ap_dict = {}
    all_ap = []

    preds = preds.cpu().numpy()   # (B, H, W)
    labels = labels.cpu().numpy() # (B, H, W)
    probs = probs.cpu().numpy()   # (B, C, H, W)

    for c in range(1, num_classes):
        # Use ALL pixels — no masking
        gt = (labels == c).ravel()                  # binary vector over all pixels
        if gt.sum() == 0:
            continue

        score = probs[:, c].ravel()                 # confidence for class c, all pixels
        # pred is not needed for ranking — we sort by score

        # Sort descending confidence
        idx = np.argsort(-score)
        gts_sorted = gt[idx]

        tp = np.cumsum(gts_sorted)
        fp = np.cumsum(1 - gts_sorted)

        total_gt = np.sum(gt)  # same as before
        recall = tp / total_gt
        precision = tp / (tp + fp + 1e-12)

        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            if np.any(recall >= t):
                p = np.max(precision[recall >= t])
                ap += p / 101

        ap_dict[c] = ap
        all_ap.append(ap)

    mAP = np.mean(all_ap) if all_ap else 0.0
    return mAP, ap_dict

def compute_macro_f1(preds, labels, num_classes=81, eps=1e-8):   # ← remove _per_class from name
    preds = preds.cpu().numpy().ravel()
    labels = labels.cpu().numpy().ravel()
    
    f1_per_class = []
    
    for c in range(1, num_classes):
        pred_c = (preds == c)
        gt_c   = (labels == c)
        
        tp = np.logical_and(pred_c, gt_c).sum()
        fp = np.logical_and(pred_c, ~gt_c).sum()
        fn = np.logical_and(~pred_c, gt_c).sum()
        
        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall + eps) if (precision + recall) > 0 else 0.0
        
        f1_per_class.append(f1)
    
    if not f1_per_class:
        return 0.0
    return np.mean(f1_per_class)

#  training loop for segmentation (uses masks, CrossEntropy, mIoU)
def trainCOCOCNN(model, num_epochs, train_loader, val_loader, optimizer, scheduler,
                 learningRate=0.001, useScheduler=False):
    """
    Training loop with:
    - Loss + mean mIoU (epoch-averaged)
    - Per-class mIoU (epoch-averaged) for train and val → saved separately
    - overall f1 NOTE: make this class based at some point...
    """
    history = {
        "train_loss": [],
        "train_miou": [],
        "val_loss": [],
        "val_miou": [],
        "lr": [],
        "val_macro_f1": []
        # "val_map": [],                    # mean AP over classes (validation)
        # "val_ap_per_class": defaultdict(list),  # class → [ap_epoch1, ap_epoch2, ...]
    }

    # Full per-class mIoU histories (class_id → list of epoch values)
    per_epoch_class_iou_train = defaultdict(list)
    per_epoch_class_iou_val   = defaultdict(list)

    for epoch in range(num_epochs):
        # ──────────────────────── Training ────────────────────────
        model.train()
        running_loss = 0.0
        running_miou = 0.0
        num_batches_train = 0

        epoch_inter_train = defaultdict(float)
        epoch_union_train = defaultdict(float)

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for images, masks, _ in train_bar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = segmentation_loss(outputs, masks, num_classes=81, dice_weight=0.5, eps=1e-8)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            # Accumulate per-class intersection/union
            for c in range(1, 81):
                p = (preds == c)
                g = (masks == c)
                inter = (p & g).sum().item()
                union = (p | g).sum().item()
                if union > 0:
                    epoch_inter_train[c] += inter
                    epoch_union_train[c] += union

            # Optional: per-batch mIoU for progress bar (still using old function)
            batch_miou = compute_miou(preds, masks, current_class_mious={}, num_classes=81, ignore_index=0)
            running_miou += batch_miou
            num_batches_train += 1

            train_bar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{batch_miou:.4f}")

        train_loss = running_loss / len(train_dataset)
        train_miou = running_miou / num_batches_train
        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        # Compute per-class mIoU for this epoch (training)
        epoch_iou_train = {}
        for c in epoch_inter_train:
            if epoch_union_train[c] > 0:
                epoch_iou_train[c] = epoch_inter_train[c] / epoch_union_train[c]

        # Store history
        for cls, iou in epoch_iou_train.items():
            per_epoch_class_iou_train[cls].append(iou)

        # ──────────────────────── Validation ───────────────────────
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        running_macro_f1 = 0.0              # ← new
        num_batches_val = 0

        epoch_inter_val = defaultdict(float)
        epoch_union_val = defaultdict(float)

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, masks, _ in val_bar:
                images = images.to(device)
                masks  = masks.to(device)

                outputs = model(images)
                loss = segmentation_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                # probs = torch.softmax(outputs, dim=1)   # ← no longer needed for F1

                # Accumulate per-class IoU
                for c in range(1, 81):
                    p = (preds == c)
                    g = (masks == c)
                    inter = (p & g).sum().item()
                    union = (p | g).sum().item()
                    if union > 0:
                        epoch_inter_val[c] += inter
                        epoch_union_val[c] += union

                # macro F1 instead of AP since AP takes a long time to calculate during training...
                batch_f1 = compute_macro_f1(preds, masks, num_classes=81)
                running_macro_f1 += batch_f1

                batch_miou = compute_miou(preds, masks, current_class_mious={}, num_classes=81, ignore_index=0)
                val_miou += batch_miou
                num_batches_val += 1

                val_bar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{batch_miou:.4f}", f1=f"{batch_f1:.4f}")

        val_loss /= len(val_dataset)
        val_miou /= num_batches_val
        val_macro_f1 = running_macro_f1 / num_batches_val

        history["val_macro_f1"].append(val_macro_f1)
        history["val_loss"].append(val_loss)         # ← good to have
        history["val_miou"].append(val_miou)

        # per-class IoU computation (unchanged)
        epoch_iou_val = {}
        for c in epoch_inter_val:
            if epoch_union_val[c] > 0:
                epoch_iou_val[c] = epoch_inter_val[c] / epoch_union_val[c]

        for cls, iou in epoch_iou_val.items():
            per_epoch_class_iou_val[cls].append(iou)

        # Scheduler & Logging
        if useScheduler and scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else learningRate
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"| train_loss: {train_loss:.4f} | train_miou: {train_miou:.4f} "
              f"| val_loss: {val_loss:.4f} | val_miou: {val_miou:.4f} "
              f"| val_macro_F1: {val_macro_f1:.4f} | lr: {current_lr:.6f}")

        # Print extremes (training)
        if epoch_iou_train:
            ranked = sorted(epoch_iou_train.items(), key=lambda x: x[1])
            shown = ranked[:5] + ranked[-5:] if len(ranked) > 10 else ranked
            parts = [f"{cls:2d}:{COCO_CLASSES[cls][:7]}{'…' if len(COCO_CLASSES[cls])>7 else ''}:{iou:.3f}"
                     for cls, iou in shown]
            line = " | ".join(parts[:5]) + (" ... " + " | ".join(parts[5:]) if len(ranked)>10 else "")
            print(f"Ep {epoch+1:3d} extremes (train) IoU: {line}")

    # ── Save histories after training ───────────────────────────────
    save_path_iou_train = os.path.join(outputFolder, "per_class_iou_train_per_epoch.pkl")
    with open(save_path_iou_train, "wb") as f:
        pickle.dump(dict(per_epoch_class_iou_train), f)

    save_path_iou_val = os.path.join(outputFolder, "per_class_iou_val_per_epoch.pkl")
    with open(save_path_iou_val, "wb") as f:
        pickle.dump(dict(per_epoch_class_iou_val), f)


    print("\nTraining finished. Saved:")
    print(f"  → {save_path_iou_train}")
    print(f"  → {save_path_iou_val}")

    return history

# plot training history
def plotTrainingResults(history, output_folder):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # ── Plot 1: Loss ────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2)
    plt.title("Cross-Entropy + Dice Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "loss_train_vs_val.png"), dpi=140)
    plt.show()

    # ── Plot 2: mIoU ────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_miou"], label="Train mIoU", linewidth=2)
    plt.plot(epochs, history["val_miou"],   label="Val mIoU",   linewidth=2)
    plt.title("Mean Intersection over Union (foreground classes)")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "miou_train_vs_val.png"), dpi=140)
    plt.show()

    # ── Plot 3: mAP (pixel-level AP) ─────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["val_macro_f1"], label="Val Macro F1", linewidth=2)
    plt.title("Macro-averaged F1 Score (foreground classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "macro_f1_val.png"), dpi=140)
    plt.show()

    # ── Learning rate (keep separate) ────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, history["lr"], label="Learning Rate", color="darkgreen", linewidth=2)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "learning_rate.png"), dpi=140)
    plt.show()
    # Drop per-class confusion (not relevant for segmentation)
# plot more class specific training history
def plot_per_class_iou_progress(output_folder, fileName="",max_classes=12,sortLargeToSmall=True):
    import pickle
    import matplotlib.pyplot as plt
    import os

    pkl_path = os.path.join(output_folder, "per_class_iou_val_per_epoch.pkl")

    
    if not os.path.exists(pkl_path):
        print("No per-class history found.")
        return

    with open(pkl_path, "rb") as f:
        history = pickle.load(f)

    if not history:
        print("No data in per-class history.")
        return

    # Sort classes by final IoU (descending)
    sorted_classes = sorted(
        history.items(),
        key=lambda x: x[1][-1] if x[1] else -1,
        reverse=sortLargeToSmall
    )

    plt.figure(figsize=(12, 7))

    plotted = 0
    for cls_id, iou_list in sorted_classes:
        if not iou_list:
            continue
        if plotted >= max_classes:
            break

        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls{cls_id}"
        epochs = list(range(1, len(iou_list) + 1))
        plt.plot(epochs, iou_list, marker=".", linewidth=1.1, label=f"{cls_id:2d} {name}")

        plotted += 1

    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Per-class mIoU progress (val) - top classes by final value" if sortLargeToSmall else "Per-class mIoU progress (val) - bottom classes by final value")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_fig_path = os.path.join(output_folder, f"{fileName}.png")
    plt.savefig(save_fig_path, dpi=140, bbox_inches="tight")
    plt.show()

    print(f"Saved plot: {save_fig_path}")

def plot_per_class_ap_progress(output_folder, max_classes=12, sortLargeToSmall=True):
    import pickle
    import matplotlib.pyplot as plt
    import os

    pkl_path = os.path.join(output_folder, "per_class_ap_per_epoch.pkl")
    if not os.path.exists(pkl_path):
        print("No per-class AP history found.")
        return

    with open(pkl_path, "rb") as f:
        history_ap = pickle.load(f)

    if not history_ap:
        print("No data in per-class AP history.")
        return

    sorted_classes = sorted(
        history_ap.items(),
        key=lambda x: x[1][-1] if x[1] else -1,
        reverse=sortLargeToSmall
    )

    plt.figure(figsize=(12, 7))
    plotted = 0
    for cls_id, ap_list in sorted_classes:
        if not ap_list:
            continue
        if plotted >= max_classes:
            break

        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls{cls_id}"
        epochs = list(range(1, len(ap_list) + 1))
        plt.plot(epochs, ap_list, marker=".", linewidth=1.1, label=f"{cls_id:2d} {name}")
        plotted += 1

    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.title("Per-class pixel AP progress (top classes by final value)" if sortLargeToSmall else "Per-class pixel AP progress (bottom classes by final value)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_fig = os.path.join(output_folder, "per_class_ap_progress.png")
    plt.savefig(save_fig, dpi=140, bbox_inches="tight")
    plt.show()

    print(f"Saved AP plot: {save_fig}")

#heatmap output of CNN decoder
def save_per_class_heatmaps(probs, images, filenames, save_dir, epoch=None, alpha=0.5,
                           min_prob_threshold=0.3, colormap='turbo', vis_threshold=0.10):
    """
    Saves a three-panel figure per class:
    - Left: original image
    - Middle: pure probability heatmap (colored mask only)
    - Right: blended overlay (image + heatmap)
    + colorbar on the far right
    """
    os.makedirs(save_dir, exist_ok=True)
    B, C, H, W = probs.shape

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(colormap)

    for b in range(B):
        fname_base = filenames[b]
        if epoch is not None:
            fname_base = f"epoch{epoch:03d}_{fname_base}"
        fname_base = os.path.splitext(fname_base)[0]

        img_np = images[b].cpu().permute(1, 2, 0).numpy()

        max_per_class = probs[b].amax(dim=(1, 2))
        active_mask = max_per_class >= min_prob_threshold
        active_classes = torch.nonzero(active_mask)[:, 0]
        max_probs_active = max_per_class[active_mask]

        if len(active_classes) == 0:
            continue

        sort_idx = torch.argsort(max_probs_active, descending=True)
        sorted_classes = active_classes[sort_idx]
        sorted_max_probs = max_probs_active[sort_idx]

        for idx, cls_tensor in enumerate(sorted_classes):
            cls = cls_tensor.item()
            if cls == 0:
                continue

            heatmap = probs[b, cls].cpu().numpy()

            # Visualization mask (for blended version only)
            alpha_mask = (heatmap >= vis_threshold).astype(np.float32)

            # Pure colored heatmap (no image underneath)
            pure_heatmap_rgb = cmap(norm(heatmap))[:, :, :3]   # (H,W,3)

            # Blended version
            blended = img_np * (1 - alpha * alpha_mask[..., None]) + \
                      pure_heatmap_rgb * (alpha * alpha_mask[..., None])
            blended = np.clip(blended, 0, 1)

            class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"cls{cls}"
            max_p = sorted_max_probs[idx].item()

            # Three-panel figure: original | pure mask | blended + colorbar
            fig = plt.figure(figsize=((W/100)*3 + 4, H/100 + 1.4))  # wider for 3 panels + colorbar

            # Left: original image
            ax_orig = fig.add_axes([0.04, 0.15, W/(W*3 + 400), 0.72])
            ax_orig.imshow(img_np)
            ax_orig.axis('off')
            ax_orig.set_title("Original", fontsize=8)

            # Middle: pure heatmap (colored probability map)
            ax_pure = fig.add_axes([0.37, 0.15, W/(W*3 + 400), 0.72])
            im_pure = ax_pure.imshow(pure_heatmap_rgb)
            ax_pure.axis('off')
            ax_pure.set_title(f"Pure {class_name} Prob", fontsize=8)

            # Right: blended overlay
            ax_blend = fig.add_axes([0.70, 0.15, W/(W*3 + 400), 0.72])
            ax_blend.imshow(blended)
            ax_blend.axis('off')
            ax_blend.set_title(f"Overlay (max: {max_p:.3f}) w/ {min_prob_threshold} thresh", fontsize=6)

            # Colorbar (far right, aligned with the panels)
            ax_cbar = fig.add_axes([0.935, 0.15, 0.018, 0.72])
            cbar = fig.colorbar(im_pure,cax=ax_cbar,orientation='vertical',ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label('Probability', rotation=270, labelpad=12, fontsize=8)

            # Overall figure title
            fig.suptitle(f"{fname_base} – Class {cls:03d} ({class_name})", fontsize=10, y=0.98)

            # Save
            save_path = os.path.join(
                save_dir,
                f"{fname_base}_class{cls:03d}_{class_name}_maxp{max_p:.3f}_three_panel.png"
            )
            fig.savefig(save_path, dpi=220, bbox_inches='tight', pad_inches=0.12)
            plt.close(fig)

    print(f"Saved three-panel figures to: {save_dir}")


# for using the model after training is complete, and saving its "outputs" given we feed it some images (heatmaps)
def predict_and_save_overlays(model,dataloader,save_dir,device,max_batches=None,epoch=None,alpha=0.5,  min_prob_threshold=0.15):
    """
    Runs inference and saves per-class heatmap overlays blended on original images
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    batch_count = 0
    
    for images, masks, filenames in tqdm(dataloader, desc="Predicting & saving class heatmaps"):
        images = images.to(device)
        
        with torch.no_grad():
            logits = model(images)                    # (B, 81, H, W)
            probs  = torch.softmax(logits, dim=1)     # probabilities

        # use data loader object to move mass around
        images_cpu = images.cpu()
        probs_cpu  = probs.cpu()
        
        save_per_class_heatmaps(probs=probs_cpu,images=images_cpu,filenames=filenames,save_dir=save_dir,epoch=epoch,alpha=alpha,min_prob_threshold=min_prob_threshold,colormap= 'turbo')
        
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break
            
    print(f"Saved per-class heatmap overlays for ≈ {batch_count * dataloader.batch_size} images")

# for handling errors in the torch DataLoader class (if images are corrupted, dont throw the whole training session just yet)
def safe_collate_fn(batch):
    # Remove items where filename starts with "SKIPPED_"
    valid = [item for item in batch if not item[2].startswith("SKIPPED_")]
    
    if not valid:
        # Very rare,  whole batch broken, return tiny dummy batch
        return (
            torch.stack([item[0] for item in batch[:1]]),   # fake batch size 1
            torch.stack([item[1] for item in batch[:1]]),
            [item[2] for item in batch[:1]]
        )
    
    return (
        torch.stack([x[0] for x in valid]),
        torch.stack([x[1] for x in valid]),
        [x[2] for x in valid]
    )

# for debugging possible image issues (solved, image 497400.png was too large with all meta data - discrarded because 1/118000 is nothing)
def dry_run_dataloader(train_loader, val_loader=None, max_batches=None):
    """
    Just iterates through the loader(s) to find which batch/image breaks it.
    No model, no loss, no training — pure data loading test.
    """
    print("Starting dry-run on train_loader...")
    batch_idx = 0
    
    try:
        for batch_idx, (images, masks, filenames) in enumerate(tqdm(train_loader, desc="Dry-run train")):
            
            # Print progress every 50 batches + always print filenames on first batch
            if batch_idx % 50 == 0 or batch_idx < 5:
                print(f"Batch {batch_idx:4d} | {len(filenames)} items | "
                      f"First few filenames: {', '.join(filenames[:3])} ...")

            # You can add shape checks here if you want
            # print(f"  image shape: {images.shape}, mask shape: {masks.shape}")

            if max_batches is not None and batch_idx >= max_batches - 1:
                print(f"Stopped after {max_batches} batches (as requested)")
                break

    except Exception as e:
        print("\n" + "="*80)
        print(f"CRASHED during train batch #{batch_idx}")
        print(f"Last successful batch had filenames like: {', '.join(filenames[:4])} ...")
        print(f"Exception: {type(e).__name__}: {str(e)}")
        print("="*80 + "\n")
        raise   # re-raise so you see full traceback

    print(f"Successfully passed ALL train batches ({batch_idx+1} batches)!")

    if val_loader is not None:
        print("\nNow dry-running validation set...")
        for batch_idx, (images, masks, filenames) in enumerate(tqdm(val_loader, desc="Dry-run val")):
            if batch_idx % 50 == 0:
                print(f"Val batch {batch_idx:4d} | {len(filenames)} items")
        print("Validation set also passed without crashing.")

    print("\nDry run FINISHED — your data loader is stable.")


def save_training_setup(output_folder, model, optimizer, scheduler, train_dataset, val_dataset):
    """
    Saves key configuration and setup info to trainingSetup.txt
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # can also add in 
    python_version = sys.version.split('\n')[0]
    torch_version = torch.__version__
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    lines = []
    lines.append("________________________________________________________________")
    lines.append(f"       Training Setup Summary - {now}")
    lines.append("----------------------------------------------------------------")
    lines.append("")
    lines.append(f"Date / Time:          {now}")
    lines.append(f"Python:               {python_version}")
    lines.append(f"PyTorch:              {torch_version}")
    lines.append(f"Device:               {device} ({device_name})")
    lines.append("")

    #  Data 
    lines.append("[ Data ]")
    lines.append(f"Image size:           {imageHeightWidth} x {imageHeightWidth}")
    lines.append(f"Train images:         {len(train_dataset):,d}")
    lines.append(f"Val images:           {len(val_dataset):,d}")
    lines.append(f"Batch size:           {batch_size}")
    lines.append(f"Augmentation (train): {augmentData}")
    lines.append(f"Cutout (train):       {train_dataset.doCutouts}")
    lines.append(f"Classes:              {model.decoder[-1].out_channels} (0=bg)")
    lines.append("")

    #  Paths 
    lines.append("[ Paths ]")
    lines.append(f"Train images:         {imageFolderTrain}")
    lines.append(f"Train masks:          {maskFolderTrain}")
    lines.append(f"Val images:           {imageFolderVal}")
    lines.append(f"Val masks:            {maskFolderVal}")
    lines.append(f"Output folder:        {output_folder}")
    lines.append("")

    #  Training 
    lines.append("[ Training ]")
    lines.append(f"Epochs:               {num_epochs}")
    lines.append(f"Optimizer:            {optimizer.__class__.__name__}")
    lines.append(f"Initial LR:           {optimizer.param_groups[0]['lr']:.2e}")
    lines.append(f"Scheduler:            {scheduler.__class__.__name__}")
    
    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        lines.append(f"  --> T_max:            {scheduler.T_max}")
        lines.append(f"  --> eta_min:          {scheduler.eta_min:.2e}")
    
    lines.append(f"Loss function:        Custom (CE + {0.5}xDice)")
    lines.append("")

    #  Model 
    lines.append("[ Model ]")
    lines.append(f"Class:                {model.__class__.__name__}")
    lines.append(f"Base channels:        {model.encoder[0].out_channels}")
    lines.append(f"Input shape:          3 x {model.expected_h} x {model.expected_w}")
    lines.append(f"Output channels:      {model.decoder[-1].out_channels}")
    
    # Optional: total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"Total parameters:     {total_params:,d}")
    lines.append(f"Trainable parameters: {trainable_params:,d}")
    lines.append("")

    #  Write to file 
    setup_path = os.path.join(output_folder, "trainingSetup.txt")
    with open(setup_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved training setup to: {setup_path}")



"""
NOTE: you need to preprocess the data using "loadCocoAndSaveLocally.py" - script should be in same folder as this one

"""
################################################ MAAAAAAAAAAAAAAAIIIIIIIIIIIIIIIIIINNNNNNNNNNNNNNNNNN

# Data and masks should be in some local folder, already down converted to some smaller and set pixel width and height 
imageHeightWidth = 64
imageFolderTrain = f"/home/npurd/trainingData/COCO/train2017_downsized{imageHeightWidth}/images"
maskFolderTrain = f"/home/npurd/trainingData/COCO/train2017_downsized{imageHeightWidth}/masks"
imageFolderVal = f"/home/npurd/trainingData/COCO/val2017_downsized{imageHeightWidth}/images"
maskFolderVal = f"/home/npurd/trainingData/COCO/val2017_downsized{imageHeightWidth}/masks"

# file names are simply the imaige ID's, and the masks have the same ID's
FileList_train = os.listdir(imageFolderTrain)
FileList_val = os.listdir(imageFolderVal)

# High level run control
num_epochs = 150
batch_size = 128 # sort of function of input image size, if 64x64 or less, do >128, if 256x256, do < 16 (exponential relationship)
learningRate = np.round(batch_size*0.00009, 6) # learning rate should be some rough function of batch size (in general, not speciifically for this NN)
augmentData = True # probably not needed for coco, we shall see (dont augment the validation data!)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


#make sure the data looks okay (make sure the data loader class is working correclty)
########################### TRAINING BELOW @##############
# set up the data loaders (note the input image size is the target size...)
train_dataset = COCOSegmentationDataset(fileList=FileList_train,image_folder=imageFolderTrain,mask_folder=maskFolderTrain,target_size=imageHeightWidth,augment=augmentData, doCutouts=False)
val_dataset = COCOSegmentationDataset(fileList=FileList_val,image_folder=imageFolderVal,mask_folder=maskFolderVal,target_size=imageHeightWidth,augment=False, doCutouts=False)
# using ~2 workers seems to be a happy spot, not sure if its cuz then more cores for training work, loading is not the bottlneck here....
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(batch_size/64), collate_fn=safe_collate_fn,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(batch_size/64), collate_fn=safe_collate_fn,pin_memory=True)


# Set up the model ()
model = RCNNSegmentation(
    num_classes=81,
    input_channels=3,
    base_channels=32,
    num_stages=3,
    kernel_sizes=[5, 3, 1],
    input_height=64,
    input_width=64
)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore background class 0
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
scheduler = CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-6,last_epoch=-1) # scheduler will change arressiveness depending on number of epochs planned on training
testName = model.__class__.__name__
outputFolder = f"testResults_{testName}{num_epochs}_{imageHeightWidth}x{imageHeightWidth}"
os.makedirs(outputFolder, exist_ok=True)
# save meta data so i can correspond output performance of DOE studies to inputs... (for mass analysis comparison)
save_training_setup(outputFolder, model, optimizer, scheduler, train_dataset, val_dataset)
# TRAIN THE THING HERE

history = trainCOCOCNN(model, num_epochs, train_loader, val_loader, optimizer, scheduler,learningRate=learningRate, useScheduler=True)
# dry_run_dataloader(train_loader, val_loader=None, max_batches=None)
# plot the results i guess
plotTrainingResults(history, outputFolder)
plot_per_class_iou_progress(outputFolder, fileName="per_class_iou_progress_best", max_classes=10, sortLargeToSmall=True) # show miou history for 10 best classes
plot_per_class_iou_progress(outputFolder,fileName="per_class_iou_progress_worst", max_classes=10,sortLargeToSmall=False)  # show miuo histoyr for worst wlasses
# plot_per_class_ap_progress(outputFolder, max_classes=10, sortLargeToSmall=True) # show MAP history
# plot_per_class_ap_progress(outputFolder, max_classes=10, sortLargeToSmall=False)
# save some predictions so we see what the heck the model is thinking (first batch of validation images)
predict_and_save_overlays(model=model,dataloader=val_loader,save_dir=os.path.join(outputFolder, "val_per_class_heatmaps"),device=device,max_batches=1,epoch=1)