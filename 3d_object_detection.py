import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ---------------- Depth Model ----------------
class DepthEstimationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, stride=2, output_padding=1, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ---------------- Device Setup ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- Load Depth Model ----------------
depth_model = DepthEstimationNet().to(device)
depth_model.load_state_dict(torch.load("./Models/NYUDEPTH.pt", map_location=device))
depth_model.eval()
print("Depth model loaded!")

# ---------------- Load Object Detection Model ----------------
num_classes = 91  # COCO
# Create model architecture
detector = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load your trained weights here
detector.load_state_dict(torch.load("./Models/coco2017.pt", map_location="cpu"))
detector.eval()
print("Object detection model loaded!")

# ---------------- COCO Labels ----------------
COCO_INSTANCE_CATEGORY_NAMES = [
     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ---------------- Transforms ----------------
depth_transform = T.Compose([
    T.Resize((240, 320)),  # match training size
    T.ToTensor(),
])

obj_transform = T.Compose([
    T.ToTensor(),
])

# ---------------- Webcam Loop ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ---- Depth Inference ----
    depth_tensor = depth_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = depth_model(depth_tensor).squeeze().cpu().numpy()

    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
    depth_tensor = depth_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = depth_model(depth_tensor).squeeze().cpu().numpy()  # original model output (H_model x W_model)

    # --- Pixelation / downsample config: change scale to increase/decrease pixel size (0.25 => 4x larger pixels)
    pixel_scale = 0.25  # 1.0 = original model resolution, <1.0 = bigger pixels (coarser)

    h_model, w_model = depth_map.shape[:2]
    small_h = max(1, int(h_model * pixel_scale))
    small_w = max(1, int(w_model * pixel_scale))

    # Create a lower-resolution depth map (float), then upscale with nearest to get blocky pixels for display
    depth_small = cv2.resize(depth_map, (small_w, small_h), interpolation=cv2.INTER_AREA)
    depth_small_up = cv2.resize(depth_small, (w_model, h_model), interpolation=cv2.INTER_NEAREST)

    # Normalize for color mapping / display
    depth_norm = cv2.normalize(depth_small_up, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
    depth_tensor = depth_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = depth_model(depth_tensor).squeeze().cpu().numpy()  # (h_model, w_model)

    
    # Pixelation / downsample config:
    # - Set pixel_scale = 1.0 for no pixelation (small pixels / full res)
    # - Use values <1.0 to increase block size (more pixelated)
    pixel_scale = 0.9  # try 1.0 (no pixelation), 0.9 (mild), 0.75, 0.5 (coarser)

    h_model, w_model = depth_map.shape[:2]
    small_h = max(1, int(h_model * pixel_scale))
    small_w = max(1, int(w_model * pixel_scale))

    # Downsample then upsample with nearest to get block appearance (if pixel_scale < 1.0)
    depth_small = cv2.resize(depth_map, (small_w, small_h), interpolation=cv2.INTER_AREA)
    depth_small_up = cv2.resize(depth_small, (w_model, h_model), interpolation=cv2.INTER_NEAREST)

    # Normalize for color mapping / display
    depth_norm = cv2.normalize(depth_small_up, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # ---- Object Detection ----
    obj_tensor = obj_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = detector(obj_tensor)[0]

    threshold = 0.6
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.int().tolist()
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]

            # Depth at center of bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                obj_depth = depth_map[cy, cx]
                depth_text = f"{obj_depth:.2f}"  # in raw units (could be meters if trained that way)
            else:
                depth_text = "N/A"
                
            # Depth at center of bounding box -> map camera coords to the downsampled depth grid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center in camera/frame coords
            # Map frame coords -> model depth map coords (w_model x h_model), then to the small grid (small_w x small_h)
            frame_h, frame_w = frame.shape[:2]
            # model grid size (before pixelation)
            # h_model, w_model defined earlier where depth_map was created
            mx = int(cx * (w_model / frame_w))
            my = int(cy * (h_model / frame_h))
            # map into small (downsampled) coordinates
            sx = int(mx * (small_w / w_model))
            sy = int(my * (small_h / h_model))
            sx = np.clip(sx, 0, small_w - 1)
            sy = np.clip(sy, 0, small_h - 1)
            obj_depth = float(depth_small[sy, sx])
            depth_text = f"{obj_depth:.2f}"
            
            if 0 <= sy < depth_small.shape[0] and 0 <= sx < depth_small.shape[1]:
                obj_depth = float(depth_small[sy, sx])
                depth_text = f"{obj_depth:.2f}"
            else:
                depth_text = "N/A"

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f} d:{depth_text}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # ---- Show combined ----
    combined = np.hstack((frame, depth_resized))
    cv2.imshow("Objects + Depth (Left) | Depth Map (Right)", combined)

    # Exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

