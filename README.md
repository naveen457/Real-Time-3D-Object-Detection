# 🚀 Real-Time 3D Object Detection (Depth + Faster R-CNN)

This project performs **3D-aware object detection** using a combination of monocular **depth estimation** and **2D object detection**. Depth cues enable approximate distance prediction in real-time from a single RGB camera.

---

## 🧩 Core Components
| Module | Purpose |
|--------|---------|
| `3d_object_detection.py` | Fuses depth + Faster R-CNN detections |
| Faster R-CNN (ResNet50-FPN) | Object classification & bounding boxes (COCO 2017) |
| DepthNet | Custom CNN encoder-decoder for monocular depth |

---

## ✨ Features
- ✔ Monocular depth estimation using NYU Depth dataset
- ✔ 2D object detection trained on COCO 2017 dataset
- ✔ Depth-fused 3D bounding box positioning
- ✔ Real-time webcam inference
- ✔ Visualization of RGB + depth + object labels

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/naveen457/Real-Time-3D-Object-Detection.git
cd Real-Time-3D-Object-Detection
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run Demo
```bash
python 3d_object_detection.py
```

### 📁 File Structure
├── README.md
├── Object_detection.py
├── Depth_estimation.py
├── 3d_object_detection.py           
├── Training/      
│   ├── coco2017.ipynb      
│   └── Nyudepth_training_colab.ipynb (This is colab google code)     
└── requirements.txt  

### 🧠 Model Architecture
🔹 DepthNet (Monocular Depth)

3-layer CNN Encoder

3-layer Transposed CNN Decoder

Trained on NYU Depth V2

Loss: Scale-invariant Depth Loss

🔹 Object Detector

Faster R-CNN (ResNet50-FPN)

Trained on COCO 2017

Outputs class + 2D bounding box

### ⚠️ Important Note

This project is trained on only 100 images for 20 epochs — purely for validating the architecture workflow (DepthNet + Faster R-CNN integration).
For better accuracy and real-world performance:

Full training scripts are included in this repository 🔧

You can train with more epochs and complete datasets

After proper training, the model can be deployed for real-time applications 🚀
