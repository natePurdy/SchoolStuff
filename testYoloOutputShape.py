# testYoloOutputShape.py
import torch
import sys

sys.path.insert(0, '/home/npurd/yolov5')

model = torch.hub.load(
    '/home/npurd/yolov5',
    'custom',
    path='/home/npurd/NN_MODELS/yolov5/yolov5l-seg.pt',
    source='local'
)

# Unwrap to get the actual Detection/SegmentationModel
if hasattr(model, 'model'):
    raw_model = model.model  # this is the DetectionModel or SegmentationModel
else:
    raw_model = model  # fallback if no wrapper

print("Raw model type:", type(raw_model))
print("Raw model nc:", getattr(raw_model, 'nc', 'Not found'))

# Final conv layer (class branch) - for seg models, look in cv3
detect_head = raw_model.model[-1] if hasattr(raw_model, 'model') else raw_model[-1]
print("Detect head type:", type(detect_head))

# Try different possible locations for class conv
if hasattr(detect_head, 'cv3'):
    print("cv3 exists")
    if isinstance(detect_head.cv3, list) or isinstance(detect_head.cv3, torch.nn.ModuleList):
        for i, branch in enumerate(detect_head.cv3):
            if hasattr(branch, 'conv'):
                print(f"cv3 branch {i} out_channels: {branch.conv.out_channels}")
            else:
                print(f"cv3 branch {i}: no conv found")
    else:
        print("cv3 out_channels:", detect_head.cv3.conv.out_channels if hasattr(detect_head.cv3, 'conv') else 'no conv')
elif hasattr(detect_head, 'cv4'):
    print("cv4 exists (mask branch):", detect_head.cv4.conv.out_channels if hasattr(detect_head.cv4, 'conv') else 'no conv')
else:
    print("No cv3/cv4 found - head structure may be different")

# Dummy inference to see raw output shape
dummy = torch.zeros((1, 3, 640, 640)).to(next(raw_model.parameters()).device)
pred = raw_model(dummy)
print("Raw prediction type:", type(pred))
print("Raw prediction length:", len(pred) if isinstance(pred, (list, tuple)) else "not list")
if isinstance(pred, (list, tuple)) and len(pred) > 0:
    print("First prediction shape:", pred[0].shape)