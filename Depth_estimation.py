import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image   # <-- Fix here

# ---- 1. Define the same model ----
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

# ---- 2. Device setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---- 3. Load model ----
model = DepthEstimationNet().to(device)
model.load_state_dict(torch.load("NYUDEPTH.pt", map_location=device))
model.eval()

print("Model loaded successfully!")

# ---- 4. Preprocessing ----
transform = T.Compose([
    T.Resize((240, 320)),  # Match training size
    T.ToTensor(),
])

# ---- 5. Open webcam ----
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (BGR -> RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        depth_map = model(input_tensor)

    # Convert depth to numpy
    depth_map_np = depth_map.squeeze().cpu().numpy()

    # Normalize depth for display
    depth_norm = cv2.normalize(depth_map_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # Apply a colormap (for better visualization)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # Resize depth map to match webcam frame
    depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # Show original + depth side by side
    combined = np.hstack((frame, depth_resized))
    cv2.imshow("Webcam (Left) | Depth Map (Right)", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
