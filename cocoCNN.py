import pickle
from collections import defaultdict
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import requests
from io import BytesIO
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
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
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
# import pandas as pd  # Uncomment if you use it elsewhere; not needed here now
# import seaborn as sns  # Drop this since no more confusion matrices

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


# use this residul block class to add resnet to an existing CNN model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels  # default same channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut for channel change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = torch.relu(out)
        return out


class CNNSegmentation(nn.Module):
    def __init__(self, num_classes=81, base_channels=32, channelDepth=3, input_height=64, input_width=64):
        super().__init__()

        if input_height % 8 != 0 or input_width % 8 != 0:
            raise ValueError(f"Input spatial size must be divisible by 8 (got {input_height}x{input_width})")

        self.expected_h = input_height
        self.expected_w = input_width
        # rgb has channel depth of 3
        c1 = base_channels          # 32
        c2 = c1 * 2                 # 64
        c3 = c2 * 2                 # 128

        self.encoder = nn.Sequential(
            nn.Conv2d(channelDepth, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(),

            ResidualBlock(c1),
            ResidualBlock(c1),
            nn.MaxPool2d(2),          #  32x32

            nn.Conv2d(c1, c2, 1),
            ResidualBlock(c2),
            ResidualBlock(c2),
            nn.MaxPool2d(2),          #  16x16

            nn.Conv2d(c2, c3, 1),
            ResidualBlock(c3),
            ResidualBlock(c3),
            nn.MaxPool2d(2),          #  8x8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c3, 2, stride=2),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            ResidualBlock(c3),

            nn.ConvTranspose2d(c3, c2, 2, stride=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            ResidualBlock(c2),

            nn.ConvTranspose2d(c2, c1, 2, stride=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),

            nn.Conv2d(c1, num_classes, 1)
        )

    def forward(self, x):
        if x.shape[2] != self.expected_h or x.shape[3] != self.expected_w:
            raise RuntimeError(
                f"Expected input shape ...x{self.expected_h}x{self.expected_w}, "
                f"got {x.shape}"
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

# Function to compute mean IoU ( "grading" metric for mask intersection)
def compute_miou(preds, labels, num_classes=81, ignore_index=0):
    """
    Compute mean IoU **only over foreground classes** (ignore background)
    preds, labels: (B, H, W) long tensors with class IDs
    """
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            continue  # skip background
        
        pred_mask = (preds == c)
        gt_mask   = (labels == c)
        
        intersection = (pred_mask & gt_mask).sum().float()
        union        = (pred_mask | gt_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
        # else: class not present -> we can skip or treat as 0/1 depending on convention
    
    if len(ious) == 0:
        return 0.0
    return np.mean(ious)

#  training loop for segmentation (uses masks, CrossEntropy, mIoU)
def trainCOCOCNN(model, num_epochs, train_loader, val_loader, optimizer):
    failed = 0 # for tracking failures to load in...
    history = {
        "train_loss": [],
        "train_miou": [],  
        "val_loss": [],
        "val_miou": []
    }


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_miou = 0.0
        num_batches_train = 0

        # try:
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)


        for images, masks, _ in train_bar:
            images = images.to(device)
            masks  = masks.to(device)


            optimizer.zero_grad()
            outputs = model(images)
            loss = segmentation_loss(outputs, masks, num_classes=81, dice_weight=0.5, eps=1e-8)
            loss.backward()
            optimizer.step()

            # Loss accumulation
            running_loss += loss.item() * images.size(0)

            # mIoU per batch
            preds = torch.argmax(outputs, dim=1)

            # if num_batches_train % 100 == 0: # for debugging
            #     unique_pred = torch.unique(preds[0]).cpu().tolist()   # first image in batch
            #     unique_maskClasses   = torch.unique(masks[0]).cpu().tolist()
                
                # print(f"Batch {num_batches_train:4d} | Pred classes: {unique_pred} | GT classes: {unique_maskClasses} | Pred fraction class 1: {(preds[0] == 1).float().mean().item()*100:.1f}%")
            batch_miou = compute_miou(preds, masks, num_classes=81, ignore_index=0)
            running_miou += batch_miou
            num_batches_train += 1

            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "miou": f"{batch_miou:.4f}",           # per-batch view
            })

        train_loss = running_loss / len(train_dataset)
        train_miou = running_miou / num_batches_train   #average miou based on number of batches in training set (its already averaged)

        # validation time
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        num_batches_val = 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks  = masks.to(device)

                outputs = model(images)
                loss = segmentation_loss(outputs, masks)

                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                batch_miou = compute_miou(preds, masks, num_classes=81, ignore_index=0)
                val_miou += batch_miou
                num_batches_val += 1

        val_loss /= len(val_dataset)
        val_miou /= num_batches_val

        # save values
        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        print(f"Epoch {epoch+1}/{num_epochs} "
            f"- train_loss: {train_loss:.4f}  "
            f"- train_miou: {train_miou:.4f}  "  
            f"- val_loss: {val_loss:.4f} "
            f"- val_miou: {val_miou:.4f}")
        # except:
        #     failed += 1
        #     print(f"failed to process/train on {failed} images")

    return history

# CHANGED: Updated plotting (drop multi-label stuff, focus on loss/mIoU)
def plotTrainingResults(history):
    # --- Plot training history ---
    # history_df = pd.DataFrame(history)  # Drop pandas if not needed
    # history_df.plot(figsize=(8,5))
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["val_miou"], label="Val mIoU")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{outputFolder}/training.png")
    plt.show()

    # Drop per-class confusion (not relevant for segmentation)


#heatmap output of CNN decoder
def save_per_class_heatmaps(probs, images, filenames,save_dir,epoch=None,alpha=0.5,min_prob_threshold=0.15,colormap='hot'):
    """
    Saves one blended heatmap overlay per class per image:
    - Heatmap intensity = model probability (0 to 1)
    - Thermal-style colormap overlaid semi-transparently on the original image
    - Only for classes with at least some meaningful confidence determined by threshold input param
    """
    os.makedirs(save_dir, exist_ok=True)
    B, C, H, W = probs.shape

    for b in range(B):
        fname_base = filenames[b]
        if epoch is not None:
            fname_base = f"epoch{epoch:03d}_{fname_base}"
        fname_base = os.path.splitext(fname_base)[0]

        # Original image as numpy (H,W,3), 0-1
        img_np = images[b].cpu().permute(1, 2, 0).numpy()

        # Max probability per class across spatial dimensions
        max_per_class = probs[b].amax(dim=(1, 2))   # use .amax() for multi-dim max
        active_classes = torch.where(max_per_class >= min_prob_threshold)[0]

        for cls in active_classes:
            if cls == 0:  # skip background
                continue

            # Get probability map for this class (H,W)
            heatmap = probs[b, cls].cpu().numpy()  # 0–1

            # Create colored heatmap using matplotlib colormap
            fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
            im = ax.imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
            ax.axis('off')
            fig.tight_layout(pad=0, h_pad=0, w_pad=0)
            fig.canvas.draw()

            # heatmap
            heatmap_norm = heatmap  # already 0-1
            colored_heatmap = cm.hot(heatmap_norm)[:, :, :3]  # (H,W,3) float RGB
            colored_heatmap = colored_heatmap.astype(np.float32)

            # Inside the class loop  replace the whole fig/canvas block:
            heatmap_norm = heatmap  # already 0-1
            cmap_func = cm.get_cmap(colormap)          # 'hot', 'inferno', etc.
            heatmap_rgb = cmap_func(heatmap_norm)[:, :, :3]  # (H, W, 3) float32 RGB, 0-1

            # No figure, no canvas, no close needed!
            # Directly blend:
            blended = img_np * (1 - alpha) + heatmap_rgb * alpha
            blended_tensor = torch.from_numpy(blended).permute(2, 0, 1).clamp(0, 1)
            class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"cls{cls}"
            # Save...
            save_path = os.path.join(
                save_dir,
                f"{fname_base}_class{cls:03d}_{class_name}_heatmap_overlay.png"
            )
            save_image(blended_tensor, save_path)

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
        
        save_per_class_heatmaps(probs=probs_cpu,images=images_cpu,filenames=filenames,save_dir=save_dir,epoch=epoch,alpha=alpha,min_prob_threshold=min_prob_threshold,colormap= 'hot')
        
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
num_epochs = 20
batch_size = 128
augmentData = False # probably not needed for coco, we shall see (dont augment the validation data!)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


#make sure the data looks okay (make sure the data loader class is working correclty)
########################### TRAINING BELOW @##############
# set up the data loaders (note the input image size is the target size...)
train_dataset = COCOSegmentationDataset(fileList=FileList_train,image_folder=imageFolderTrain,mask_folder=maskFolderTrain,target_size=imageHeightWidth,augment=augmentData, doCutouts=False)
val_dataset = COCOSegmentationDataset(fileList=FileList_val,image_folder=imageFolderVal,mask_folder=maskFolderVal,target_size=imageHeightWidth,augment=False, doCutouts=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=safe_collate_fn,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=safe_collate_fn,pin_memory=True)


# Set up the model ()
model = CNNSegmentation(num_classes=81, base_channels=32, channelDepth=3, input_height=imageHeightWidth, input_width=imageHeightWidth) 
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore background class 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
testName = model.__class__.__name__
outputFolder = f"testResults_{testName}"
os.makedirs(outputFolder, exist_ok=True)
# TRAIN THE THING HERE
history = trainCOCOCNN(model, num_epochs, train_loader, val_loader, optimizer)
# dry_run_dataloader(train_loader, val_loader=None, max_batches=None)

# save some predictions so we see what the heck the model is thinking (first batch of validation images)
predict_and_save_overlays(model=model,dataloader=val_loader,save_dir=os.path.join(outputFolder, "val_per_class_heatmaps"),device=device,max_batches=1,epoch=1)
# plot the results i guess
plotTrainingResults(history)