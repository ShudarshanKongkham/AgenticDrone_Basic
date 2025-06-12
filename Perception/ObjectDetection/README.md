# 🎯 State-of-the-Art Live Object Detector

A comprehensive real-time object detection system powered by **YOLOv8** (You Only Look Once version 8), featuring advanced AI capabilities, modern GUI interface, and professional-grade performance monitoring.

![Object Detection](https://img.shields.io/badge/AI-YOLOv8-blue)
![Real_Time](https://img.shields.io/badge/Real--Time-Live%20Detection-green)
![Performance](https://img.shields.io/badge/Performance-State--of--the--Art-orange)

## ✨ Features

### 🧠 Advanced AI Models
- **YOLOv8 Nano (yolov8n.pt)** - Ultra-fast detection for real-time applications
- **YOLOv8 Small (yolov8s.pt)** - Balanced speed and accuracy
- **YOLOv8 Medium (yolov8m.pt)** - High accuracy for demanding applications
- **YOLOv8 Large (yolov8l.pt)** - Very high accuracy for professional use
- **YOLOv8 Extra Large (yolov8x.pt)** - Maximum accuracy for research/production

### 🎥 Video Input Sources
- **Live Webcam Feed** - Real-time detection from camera
- **Video Files** - Process MP4, AVI, MOV, MKV, FLV, WMV files
- **High Resolution Support** - Up to 1280x720 for optimal performance
- **Auto-resolution Adjustment** - Intelligent frame sizing

### ⚙️ Advanced Configuration
- **Confidence Threshold** - Adjustable from 0.1 to 0.95
- **IoU (Intersection over Union) Threshold** - Fine-tune detection overlap
- **Maximum Detections** - Control processing load (up to 300 objects)
- **Real-time Parameter Adjustment** - No need to restart

### 📊 Performance Analytics
- **Real-time FPS Monitoring** - Live frame rate display
- **Detection Statistics** - Total counts, confidence averages
- **Class Distribution Analysis** - Object type frequency tracking
- **Performance Metrics Export** - JSON format for analysis

### 🎮 Modern GUI Interface
- **Live Video Feed Display** - 800x600 optimized viewer
- **Interactive Controls** - Intuitive button layout
- **Real-time Status Updates** - Visual feedback system
- **Professional Dark Theme** - Easy on the eyes

### 📹 Recording & Export
- **Live Recording** - Save detection sessions to MP4
- **Frame Capture** - Save individual frames with detections
- **Statistics Export** - Comprehensive analytics in JSON
- **Timestamped Outputs** - Organized file naming

## 🚀 Quick Start

### 1. Installation
```bash
# Install required dependencies
pip install ultralytics opencv-python pillow numpy

# Or use the project requirements
pip install -r requirements.txt
```

### 2. Launch the Detector
```bash
# Method 1: Direct launch
python ObjectDetection/detectObject.py

# Method 2: Using launcher
python launch_object_detector.py
```

### 3. Using the Interface

1. **Select AI Model** - Choose from YOLOv8 variants based on your needs:
   - `Nano` for maximum speed (30+ FPS)
   - `Small` for balanced performance (20-30 FPS)
   - `Medium` for high accuracy (15-25 FPS)
   - `Large` for professional use (10-20 FPS)
   - `Extra Large` for research/maximum accuracy (5-15 FPS)

2. **Adjust Parameters**:
   - **Confidence**: Lower values detect more objects (may include false positives)
   - **IoU**: Higher values reduce duplicate detections

3. **Start Detection**:
   - Click `📷 Start Webcam` for live camera feed
   - Click `📁 Load Video File` to process video files

4. **Monitor Performance**:
   - Watch real-time FPS in the stats panel
   - Track detection counts and accuracy

5. **Record & Export**:
   - Use `⏺️ Start Recording` to save detection sessions
   - Export analytics with `📈 Export Statistics`

## 📈 Performance Benchmarks

### Model Comparison (on typical hardware)

| Model | Speed (FPS) | Accuracy (mAP) | Use Case |
|-------|-------------|----------------|----------|
| YOLOv8n | 30-50 | 37.3 | Real-time applications |
| YOLOv8s | 20-35 | 44.9 | Balanced performance |
| YOLOv8m | 15-25 | 50.2 | High-accuracy apps |
| YOLOv8l | 10-20 | 52.9 | Professional use |
| YOLOv8x | 5-15 | 53.9 | Maximum accuracy |

### Supported Object Classes (80+ COCO Classes)
- **People & Animals**: person, cat, dog, horse, cow, etc.
- **Vehicles**: car, truck, bus, motorcycle, bicycle, etc.
- **Everyday Objects**: chair, table, laptop, phone, book, etc.
- **Sports Equipment**: ball, bat, racket, surfboard, etc.
- **And many more...**

## 🔧 Technical Specifications

### System Requirements
- **Python**: 3.7+ (3.8+ recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: Optional but recommended for faster inference
- **Storage**: 2GB for models and dependencies

### Dependencies
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python>=4.5.0` - Computer vision operations
- `pillow>=8.0.0` - Image processing
- `numpy>=1.20.0` - Numerical operations
- `tkinter` - GUI framework (included with Python)

### Input Formats Supported
- **Images**: JPG, PNG, BMP, TIFF
- **Videos**: MP4, AVI, MOV, MKV, FLV, WMV
- **Live Streams**: Webcam, IP cameras, RTSP streams

## 🎯 Advanced Usage

### Programmatic Usage
```python
from detectObject import StateOfTheArtObjectDetector

# Create detector instance
detector = StateOfTheArtObjectDetector()

# Configure parameters
detector.confidence_threshold = 0.7
detector.model_name = "yolov8m.pt"

# Run the application
detector.run()
```

### Custom Model Integration
```python
# Load custom trained model
detector.model_name = "path/to/your/custom_model.pt"
detector._load_model()
```

### Batch Processing
```python
# Process multiple video files
video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in video_files:
    detector._load_video_file(video)
    # Processing logic here
```

## 📊 Output Examples

### Statistics Export (JSON)
```json
{
  "timestamp": "2024-01-15T14:30:00",
  "model_used": "yolov8m.pt",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "total_frames_processed": 1500,
  "total_detections": 4250,
  "average_confidence": 0.73,
  "average_fps": 22.5,
  "class_distribution": {
    "person": 1200,
    "car": 800,
    "bicycle": 350,
    "dog": 150
  }
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **YOLOv8 Installation Error**
   ```bash
   pip install --upgrade ultralytics
   ```

2. **Webcam Not Detected**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Restart the application

3. **Low FPS Performance**
   - Use YOLOv8n (nano) model
   - Reduce camera resolution
   - Close other applications
   - Enable GPU acceleration if available

4. **Memory Issues**
   - Reduce max_detections parameter
   - Use smaller model variant
   - Close other applications

### GPU Acceleration
For NVIDIA GPUs with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional model formats (ONNX, TensorRT)
- Custom training pipeline integration
- Advanced analytics features
- Mobile/edge device optimization

## 📄 License

This project is part of the Agentic Drone system and follows the same licensing terms.

## 🔗 Related Projects

- **Main Project**: [Agentic Drone Control System](../README.md)
- **Depth Estimation**: [DepthEstimation Module](../DepthEstimation/)
- **Drone Integration**: [Tello Drone Package](../tello_drone/)

---

**Built with ❤️ using YOLOv8 and modern Python practices** 