# yolo5_eval_working.py
import torch
import sys
from pathlib import Path

# Add yolov5 repo to sys.path (critical!)
YOLOV5_PATH = '/home/npurd/yolov5'  # change if cloned elsewhere
sys.path.insert(0, YOLOV5_PATH)

# Import YOLOv5 modules
from models.experimental import attempt_load
from utils.general import check_requirements
from val import run as yolov5_val

def evaluate_yolov5(model_path, data_yaml="/home/npurd/coco.yaml", imgsz=640, batch=8, device=0):
    print(f"Evaluating: {model_path.name}")
    
    device_str = f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu"
    
    # Load model using local repo
    model = attempt_load(
        str(model_path),
        device=device_str,
        inplace=True,
        fuse=True
    )
    
    print("Model loaded. nc:", model.nc)
    print("Class names length:", len(model.names))
    
    model.eval()
    
    print(f"Validating on {data_yaml} (imgsz={imgsz}, batch={batch}, device={device_str})")
    
    results, _, _ = yolov5_val(
        data=data_yaml,
        weights=str(model_path),
        batch_size=batch,
        imgsz=imgsz,
        conf_thres=0.001,
        iou_thres=0.7,
        device=device_str,
        save_json=True,
        plots=True,
        project="runs/val_yolo5",
        name=Path(model_path).stem,
    )
    
    print("\nValidation complete!")
    print(f"mAP@50:95 (box): {results.box.map:.4f}")
    print(f"mAP@50 (box):    {results.box.map50:.4f}")
    if hasattr(results, 'seg'):
        print(f"mAP@50:95 (mask): {results.seg.map:.4f}")
        print(f"mAP@50 (mask):    {results.seg.map50:.4f}")

if __name__ == "__main__":
    modelDir = Path("/home/npurd/NN_MODELS/yolov5")
    
    size_order = {'n':0, 's':1, 'm':2, 'l':3, 'x':4}
    
    models = sorted(
        [p for p in modelDir.iterdir() if p.suffix == ".pt"],
        key=lambda p: size_order.get(
            next((c for c in p.stem.lower() if c in size_order), 'z'), 99
        )
    )
    
    data_yaml = "/home/npurd/coco.yaml"
    
    for model_path in models:
        evaluate_yolov5(model_path, data_yaml=data_yaml, batch=8, device=0)