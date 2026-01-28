import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------
# CONFIG
# ---------------------------
image_path = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/train/images/"        # Your downsized image
mask_path = "/mnt/d/SCHOOL_crap/ece_523/sandbox/dataSets/COCO/train/masks/"          # Corresponding mask
num_classes = 81                        # Number of classes (include background=0)

# Optional: provide a dict of class ID -> name
class_id_to_name = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}

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



listOfImageFiles = os.listdir(image_path)
listOfMaskFiles = os.listdir(mask_path)



for imageFile, maskFile in zip(listOfImageFiles, listOfMaskFiles):
    # ---------------------------
    # LOAD IMAGE AND MASK
    # ---------------------------
    imagePathFull = image_path + imageFile
    maskPathFull = mask_path + maskFile
    img = Image.open(imagePathFull).convert("RGB")
    mask = Image.open(maskPathFull)

    img_np = np.array(img)
    mask_np = np.array(mask)

    # ---------------------------
    # DISPLAY IMAGE WITH MASK
    # ---------------------------
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.title(f"Image:{imageFile} --> Mask:{maskFile}")

    # Create a colored overlay for mask
    # Generate colors for classes
    colors = plt.cm.get_cmap("tab20", num_classes)  # 20 distinct colors
    mask_colored = np.zeros_like(img_np, dtype=np.uint8)

    for class_id in np.unique(mask_np):
        if class_id == 0:
            continue

        mask_class = (mask_np == class_id)
        color = BASE_COLORS[class_id % len(BASE_COLORS)]
        mask_colored[mask_class] = color

    # Overlay with some transparency
    plt.imshow(mask_colored, alpha=0.5)

    # ---------------------------
    # LEGEND
    # ---------------------------
    handles = []
    for class_id in sorted(np.unique(mask_np)):
        if class_id == 0:
            continue

        name = class_id_to_name.get(class_id, f"ID {class_id}")
        color = BASE_COLORS[class_id % len(BASE_COLORS)] / 255.0

        patch = mpatches.Patch(color=color, label=f"{name} ({class_id})")
        handles.append(patch)
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8
    )
    plt.axis("off")
    plt.show()
