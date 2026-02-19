from ultralytics import YOLO
from pathlib import Path
import cv2
import os
from tqdm import tqdm
"""
The purpose of this script is to simply run or train a yolo segmentation model in order to get some baseline results for the 
project in ece523 this semester.

"""


INPUT_FOLDER = "/home/npurd/trainingData/COCO/val2017/"          # ← your folder with original images
OUTPUT_FOLDER = "/home/npurd/trainingData/COCO/BASELINE_yolo11n-seg_results_val2017/"  # ← where to save annotated images

model_dir = "/home/npurd/NN_MODELS/"
model_name = "yolo11x-seg"          # options: n, s, m, l, x
model_extension = ".pt"
modelPathLocal = model_dir + model_name + model_extension   # already downloaded locally to save 5 seconds every time model is called, use the local version ( only ~150 MB)
# MODEL_SIZE = "yolo11s-seg"        # better quality, a bit slower
# MODEL_SIZE = "yolo11m-seg"        # even better, noticeably slower on CPU

CONF_THRESHOLD = 0.25               # minimum confidence to show a detection
IOU_THRESHOLD = 0.45                # NMS IoU threshold

SAVE_MASKS_AS_OVERLAY = True        # True = colorful mask overlay, False = just boxes + labels
SAVE_TXT_ANNOTATIONS = False        # optional: save YOLO-format .txt labels per image

# ────────────────────────────────────────────────

def main():
    # Create output folder if it doesn't exist
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading YOLOv11-{modelPathLocal} segmentation model...")
    model = YOLO(modelPathLocal)

    # Find all image files
    image_names = os.listdir(INPUT_FOLDER)
    image_paths = []
    for image in image_names:
        image_paths.append(INPUT_FOLDER + image)

    if not image_paths:
        print(f"No images found in {INPUT_FOLDER}")
        return

    print(f"Found {len(image_paths)} images. Starting inference...\n")

    # Optional: set inference parameters
    model.overrides.update({
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "imgsz": 640,                # can be 640 or 1280 etc.
        "half": True,                # FP16 if GPU supports it
    })

    # Process images with progress bar
    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        try:
            # Run inference
            results = model(img_path, verbose=False)

            # Get first (and usually only) result
            r = results[0]

            # Get the annotated image (with boxes + masks + labels)
            annotated_img = r.plot(
                    labels=True,                # class names + conf in text
                    boxes=True,
                    masks=SAVE_MASKS_AS_OVERLAY,
                    probs=False,                # no class prob bars
                    line_width=None,            # auto
                )

            # Convert from RGB (ultralytics) to BGR (OpenCV save)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

            # Build output filename

            # Alternative: add prefix
            # out_name = f"yolov11-{MODEL_SIZE}_{img_path.name}"
            imageSaveName = img_path.split("/")[-1]

            out_path = OUTPUT_FOLDER + imageSaveName

            # Save annotated image
            cv2.imwrite(str(out_path), annotated_img)

            # Optional: save YOLO-format txt annotations
            if SAVE_TXT_ANNOTATIONS:
                txt_path = out_path.with_suffix(".txt")
                with open(txt_path, "w") as f:
                    for box, cls, conf in zip(r.boxes.xywhn, r.boxes.cls, r.boxes.conf):
                        f.write(f"{int(cls)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {conf:.6f}\n")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nDone! Annotated images saved to:")
    print(f"  {OUTPUT_FOLDER}")
    print(f"Processed {len(image_paths)} images.")

if __name__ == "__main__":
    main()