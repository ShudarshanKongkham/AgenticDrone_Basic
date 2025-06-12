#!/usr/bin/env python3
"""
Object Distance Analysis
========================

A streamlined system that combines YOLO object detection with depth estimation
to identify objects and calculate their distances from the camera.

Usage:
    python perception_analysis.py           # Single frame analysis
    python perception_analysis.py --live    # Real-time analysis
"""

import cv2
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from ultralytics import YOLO
from PIL import Image
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

@dataclass
class DetectedObject:
    """Data class for detected object with distance"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    distance: float  # in meters
    
    def __str__(self):
        return f"{self.name} ({self.confidence:.2f}) at {self.distance:.2f}m"

class ObjectDistanceAnalyzer:
    """Combines YOLO object detection with depth estimation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Load models
        self._load_models()
        print("✅ Analyzer ready!")
    
    def _load_models(self):
        """Load YOLO and depth estimation models"""
        print("📥 Loading models...")
        
        # Load YOLO (lightweight model)
        self.yolo_model = YOLO("yolo11n.pt")
        
        # Load depth model
        try:
            self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            self.depth_model.to(self.device)
            self.depth_model.eval()
        except Exception as e:
            print(f"❌ Failed to load depth model: {e}")
            raise
    
    def predict_depth(self, image):
        """Predict depth map from image"""
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Process and predict
            inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
            
            return depth_map
        except Exception as e:
            print(f"Error in depth prediction: {e}")
            return None
    
    def calibrate_depth(self, depth_map, min_dist=0.5, max_dist=8.0):
        """Convert depth to real-world distances"""
        if depth_map.min() < 0:
            depth_map = -depth_map
        
        # Normalize and map to distance range
        normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        distance_map = min_dist + (1 - normalized) * (max_dist - min_dist)
        
        return distance_map
    
    def analyze_frame(self, frame, confidence=0.5):
        """Analyze frame for objects and distances"""
        # Detect objects
        results = self.yolo_model(frame, conf=confidence, verbose=False)
        
        # Get depth
        depth_map = self.predict_depth(frame)
        if depth_map is None:
            return []
        
        # Convert to distance
        distance_map = self.calibrate_depth(depth_map)
        
        # Resize distance map to frame size
        h, w = frame.shape[:2]
        distance_map = cv2.resize(distance_map, (w, h))
        
        detected_objects = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get detection data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo_model.names[class_id]
                
                # Calculate center distance
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x = max(0, min(center_x, w - 1))
                center_y = max(0, min(center_y, h - 1))
                
                distance = float(distance_map[center_y, center_x])
                
                obj = DetectedObject(
                    name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    distance=distance
                )
                detected_objects.append(obj)
        
        return detected_objects
    
    def draw_detections(self, frame, objects):
        """Draw bounding boxes and distance labels"""
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{obj.name}: {obj.distance:.1f}m"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            
            # Label text
            cv2.putText(frame, label, (x1+5, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        return frame
    
    def single_frame_analysis(self):
        """Capture and analyze a single frame"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("📸 Capturing frame...")
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Failed to capture frame")
            return
        
        # Analyze frame
        print("🔍 Analyzing...")
        objects = self.analyze_frame(frame)
        
        # Draw results
        result_frame = self.draw_detections(frame.copy(), objects)
        
        # Display
        cv2.imshow('Object Distance Analysis - Press any key to close', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print results
        print(f"\n🎯 Found {len(objects)} objects:")
        for i, obj in enumerate(objects, 1):
            print(f"{i}. {obj}")
    
    def live_analysis(self, duration=30):
        """Real-time analysis"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print(f"🎥 Starting live analysis for {duration} seconds...")
        print("Press 'q' to quit, 's' to save frame")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                objects = self.analyze_frame(frame, confidence=0.4)
                frame = self.draw_detections(frame, objects)
            
            # Add timer
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"Time: {remaining}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Live Object Distance Analysis', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Analysis completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Object Distance Analysis')
    parser.add_argument('--live', action='store_true', 
                       help='Run live analysis instead of single frame')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration for live analysis (seconds)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ObjectDistanceAnalyzer()
        
        if args.live:
            analyzer.live_analysis(args.duration)
        else:
            analyzer.single_frame_analysis()
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("  - Working webcam")
        print("  - Required dependencies installed")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
