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


class COCOSegmentationDataset(Dataset):
    def __init__(
        self,
        fileList,
        image_folder,       # <-- Local folder
        mask_folder,        # local folder also
        target_size=256,
        augment=False,
        num_classes=81
    ):
        self.fileList = fileList
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_size = target_size
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):

        filename = self.fileList[idx]
        # --- Load image and mask 
        img_path = os.path.join(self.image_folder, filename)
        mask_path = os.path.join(self.mask_folder, filename)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # greyscale

        # --- Optional augmentation ---
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # --- To tensors ---
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (C,H,W)
        mask = np.array(mask, dtype=np.int64)


        # --- Multi-label vector ---
        unique_ids = np.unique(mask)
        multilabel = np.zeros(self.num_classes, dtype=np.float32)
        multilabel[unique_ids] = 1.0

        return img, mask, multilabel, filename


class DeeperCNNMultiLabel(nn.Module):
    """
    CNN for multi-label classification of entire images.
    Input: (B, 3, 256, 256)
    Output: (B, num_classes) with sigmoid activations
    """
    def __init__(self, input_channels=3, num_classes=81):
        super().__init__()

        # --- Feature extractor ---
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        # --- Global pooling + classifier ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # pool to (B, 128, 1, 1)
            nn.Flatten(),                  # (B, 128)
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)   # (B, num_classes)
        )

    def forward(self, x):
        x = self.features(x)           # (B, 128, 32, 32)
        x = self.classifier(x)         # (B, num_classes)
        x = torch.sigmoid(x)           # multi-label probabilities
        return x



#  Function to visualize a few images with masks 
# Function to visualize a few images with masks
def show_images_with_masks(dataset, n=5):
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

    # Normalize colors for matplotlib (0-1)
    BASE_COLORS = BASE_COLORS / 255.0


    
    for i in range(n):
        plt.figure(figsize=(12, 6))
        img_np, mask_np, _, fname = dataset[i]
        img_np = np.transpose(img_np, (1,2,0))  # (H,W,C)

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



"""
NOTE: you need to preprocess the data using "loadCocoAndSaveLocally.py" - script should be in same folder as this one

"""

# Data and masks should be in some local folder, already down converted to some smaller and set pixel width and height 
imageFolderTrain = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/train/images"
maskFolderTrain = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/train/masks"
imageFolderVal = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/val/images"
maskFolderVal = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/val/masks"

# file names are simply the imaige ID's, and the masks have the same ID's
FileList_train = os.listdir(imageFolderTrain)
FileList_val = os.listdir(imageFolderVal)


# instantiate the dataset classes
train_dataset = COCOSegmentationDataset(fileList=FileList_train, image_folder=imageFolderTrain, mask_folder=maskFolderTrain, target_size=256, augment=True)
val_dataset = COCOSegmentationDataset(fileList=FileList_val, image_folder=imageFolderVal, mask_folder=maskFolderVal, target_size=256, augment=True)

# check n number of images if you are worried about the data loader like i am
# show_images_with_masks(train_dataset, n=5)

# set up the data loading paradigm of the data set classes
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)


#make sure the data looks okay (make sure the data loader class is working correclty)
########################### TRAINING BELOW @##############
# ---------------------------
# --- Settings ---
# ---------------------------
num_epochs = 10
batch_size = 16
threshold = 0.5  # for pixel-wise prediction
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# --- Instantiate dataset & loader ---
# ---------------------------
train_dataset = COCOSegmentationDataset(
    fileList=FileList_train,
    image_folder=imageFolderTrain,
    mask_folder=maskFolderTrain,
    target_size=256,
    augment=True
)
val_dataset = COCOSegmentationDataset(
    fileList=FileList_val,
    image_folder=imageFolderVal,
    mask_folder=maskFolderVal,
    target_size=256,
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------------
# --- Model, criterion, optimizer ---
# ---------------------------
# --- Model and optimizer ---
model = DeeperCNNMultiLabel(input_channels=3, num_classes=81).to(device)
criterion = nn.BCELoss()  # outputs are sigmoid probabilities
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training history ---
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for images, masks, multilabels, _ in train_bar:
        images = images.to(device)
        labels = multilabels.to(device)  # shape: (B, num_classes)

        optimizer.zero_grad()
        outputs = model(images)           # shape: (B, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

       # --- Multi-label accuracy per class, ignoring absent classes and background ---
        preds = (outputs > 0.5).float()

        # Mask of classes to consider: present in image AND not background (class 0)
        present_mask = (labels == 1) & (torch.arange(labels.size(1), device=device) != 0).unsqueeze(0)
        # Explanation:
        # - labels == 1 → only classes present in each image
        # - torch.arange(...) != 0 → ignore class 0 (background)
        # - unsqueeze(0) broadcasts across batch dimension

        # Count correct predictions only for these classes
        running_correct += ((preds == labels) & present_mask).sum().item()
        running_total += present_mask.sum().item()


        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{running_correct/running_total:.4f}"  # per-class accuracy ignoring absent classes
        })

    train_loss = running_loss / len(train_dataset)
    train_acc = running_correct / running_total

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, masks, multilabels, _ in val_loader:
            images = images.to(device)
            labels = multilabels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            present_mask = labels == 1

            val_correct += ((preds == labels) & present_mask).sum().item()
            val_total += present_mask.sum().item()

    val_loss /= len(val_dataset)
    val_acc = val_correct / val_total

    # --- Save history ---
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"- train_loss: {train_loss:.4f} "
        f"- train_acc: {train_acc:.4f} "
        f"- val_loss: {val_loss:.4f} "
        f"- val_acc: {val_acc:.4f}"
    )


# --- Plot training history ---
history_df = pd.DataFrame(history)
history_df.plot(figsize=(8,5))
plt.grid(True)
plt.show()

# per clas confusion
import seaborn as sns
num_classes = preds_all.shape[1]
class_names = COCO_CLASSES[:num_classes]

# Compute per-class TP, FP, FN, TN
conf_matrices = []
for c in range(num_classes):
    TP = ((labels_all[:,c]==1) & (preds_all[:,c]==1)).sum()
    TN = ((labels_all[:,c]==0) & (preds_all[:,c]==0)).sum()
    FP = ((labels_all[:,c]==0) & (preds_all[:,c]==1)).sum()
    FN = ((labels_all[:,c]==1) & (preds_all[:,c]==0)).sum()
    conf_matrices.append([[TP, FP],[FN, TN]])

# Optional: plot per-class heatmaps
for c, cm in enumerate(conf_matrices):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 1","Pred 0"], yticklabels=["True 1","True 0"])
    plt.title(f"Confusion matrix for class: {class_names[c]}")
    plt.show()
