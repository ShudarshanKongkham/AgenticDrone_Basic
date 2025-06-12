#!/usr/bin/env python3
"""
State-of-the-Art Live Object Detector
=====================================

A comprehensive real-time object detection system using YOLO11, featuring:
- Live video feed processing from webcam or video files
- Modern GUI interface with real-time controls
- Advanced detection models (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x)
- Performance metrics and FPS monitoring
- Configurable confidence thresholds and IoU parameters
- Export capabilities for detected objects and analytics
- Support for custom classes and model switching

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from pathlib import Path
import json
from datetime import datetime
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Installing...")

class StateOfTheArtObjectDetector:
    """Advanced real-time object detection system with YOLO11."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 State-of-the-Art Live Object Detector")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Detection parameters
        self.model = None
        self.model_name = "yolo11n.pt"  # Default lightweight model
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 300
        
        # Video processing
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.current_frame = None
        self.processed_frame = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        self.detection_stats = {
            'total_detections': 0,
            'avg_confidence': 0.0,
            'class_counts': {}
        }
        
        # GUI components
        self.video_label = None
        self.status_var = tk.StringVar(value="🔴 Stopped")
        self.fps_var = tk.StringVar(value="FPS: 0")
        self.detection_var = tk.StringVar(value="Detections: 0")
        
        self._setup_gui()
        self._load_model()
        
    def _setup_gui(self):
        """Create the modern GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for video
        left_panel = ttk.LabelFrame(main_frame, text="🎥 Live Detection Feed", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video display
        self.video_label = tk.Label(left_panel, bg='black', text="📷\nNo video feed\nClick 'Start Webcam' to begin")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for controls
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Model selection
        model_frame = ttk.LabelFrame(right_panel, text="🧠 AI Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_var = tk.StringVar(value="yolo11n.pt")
        models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        model_descriptions = {
            "yolo11n.pt": "Nano - Ultra Fast",
            "yolo11s.pt": "Small - Balanced",
            "yolo11m.pt": "Medium - High Accuracy",
            "yolo11l.pt": "Large - Very High Accuracy",
            "yolo11x.pt": "Extra Large - Maximum Accuracy"
        }
        
        for model in models:
            ttk.Radiobutton(
                model_frame, 
                text=model_descriptions[model], 
                variable=self.model_var, 
                value=model,
                command=self._change_model
            ).pack(anchor=tk.W)
        
        # Detection parameters
        params_frame = ttk.LabelFrame(right_panel, text="⚙️ Detection Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(params_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(
            params_frame, 
            from_=0.1, 
            to=0.95, 
            variable=self.conf_var, 
            orient=tk.HORIZONTAL,
            command=self._update_confidence
        )
        conf_scale.pack(fill=tk.X, pady=(0, 5))
        self.conf_label = ttk.Label(params_frame, text="0.5")
        self.conf_label.pack(anchor=tk.W)
        
        # IoU threshold
        ttk.Label(params_frame, text="IoU Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_scale = ttk.Scale(
            params_frame, 
            from_=0.1, 
            to=0.95, 
            variable=self.iou_var, 
            orient=tk.HORIZONTAL,
            command=self._update_iou
        )
        iou_scale.pack(fill=tk.X, pady=(0, 5))
        self.iou_label = ttk.Label(params_frame, text="0.45")
        self.iou_label.pack(anchor=tk.W)
        
        # Control buttons
        control_frame = ttk.LabelFrame(right_panel, text="🎮 Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Video source buttons
        ttk.Button(
            control_frame, 
            text="📷 Start Webcam", 
            command=self._start_webcam,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            control_frame, 
            text="📁 Load Video File", 
            command=self._load_video_file
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            control_frame, 
            text="⏹️ Stop Detection", 
            command=self._stop_detection
        ).pack(fill=tk.X, pady=2)
        
        # Recording controls
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Button(
            control_frame, 
            text="⏺️ Start Recording", 
            command=self._start_recording
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            control_frame, 
            text="⏹️ Stop Recording", 
            command=self._stop_recording
        ).pack(fill=tk.X, pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="📊 Performance Stats", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(stats_frame, textvariable=self.status_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.fps_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.detection_var).pack(anchor=tk.W)
        
        # Export buttons
        export_frame = ttk.LabelFrame(right_panel, text="💾 Export & Analytics", padding=10)
        export_frame.pack(fill=tk.X)
        
        ttk.Button(
            export_frame, 
            text="📈 Export Statistics", 
            command=self._export_statistics
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            export_frame, 
            text="🖼️ Save Current Frame", 
            command=self._save_current_frame
        ).pack(fill=tk.X, pady=2)
        
    def _load_model(self):
        """Load the YOLO model."""
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "YOLO11 not available. Please install ultralytics: pip install ultralytics")
            return
            
        try:
            print(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            print(f"✅ Model {self.model_name} loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            
    def _change_model(self):
        """Change the detection model."""
        new_model = self.model_var.get()
        if new_model != self.model_name:
            self.model_name = new_model
            self._load_model()
            
    def _update_confidence(self, value):
        """Update confidence threshold."""
        self.confidence_threshold = float(value)
        self.conf_label.config(text=f"{self.confidence_threshold:.2f}")
        
    def _update_iou(self, value):
        """Update IoU threshold."""
        self.iou_threshold = float(value)
        self.iou_label.config(text=f"{self.iou_threshold:.2f}")
        
    def _start_webcam(self):
        """Start webcam detection."""
        if self.is_running:
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam")
            return
            
        # Set camera resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.status_var.set("🟢 Running - Webcam")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
    def _load_video_file(self):
        """Load video file for detection."""
        if self.is_running:
            self._stop_detection()
            
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Video Error", "Could not open video file")
                return
                
            self.is_running = True
            self.status_var.set(f"🟢 Running - {Path(file_path).name}")
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
    def _stop_detection(self):
        """Stop the detection process."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set("🔴 Stopped")
        
    def _start_recording(self):
        """Start recording detected video."""
        if not self.is_running:
            messagebox.showwarning("Recording", "Start detection first")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"detected_video_{timestamp}.mp4"
        
        # Get frame dimensions
        if self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, 20.0, (width, height)
            )
            self.is_recording = True
            messagebox.showinfo("Recording", f"Started recording to {self.output_path}")
            
    def _stop_recording(self):
        """Stop recording."""
        if self.is_recording:
            self.is_recording = False
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
            messagebox.showinfo("Recording", f"Recording saved to {self.output_path}")
            
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        self.fps_start_time = time.time()
        self.fps_counter = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            
            # Perform detection
            if self.model:
                try:
                    results = self.model(
                        frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_detections,
                        verbose=False
                    )
                    
                    # Process results
                    self.processed_frame = self._draw_detections(frame, results[0])
                    self._update_statistics(results[0])
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                    self.processed_frame = frame
            else:
                self.processed_frame = frame
                
            # Update FPS
            self._update_fps()
            
            # Record if needed
            if self.is_recording and hasattr(self, 'video_writer'):
                self.video_writer.write(self.processed_frame)
                
            # Update GUI
            self.root.after(1, self._update_video_display)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
        self._stop_detection()
        
    def _draw_detections(self, frame, results):
        """Draw detection boxes and labels on frame."""
        if results.boxes is None:
            return frame
            
        # Get detection data
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Class names from COCO dataset
        class_names = results.names
        
        # Colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class info
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
        # Add FPS and detection count overlay
        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f} | Detections: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame
        
    def _update_statistics(self, results):
        """Update detection statistics."""
        if results.boxes is not None:
            num_detections = len(results.boxes)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            class_names = results.names
            
            # Update total counts
            self.detection_stats['total_detections'] += num_detections
            
            # Update average confidence
            if num_detections > 0:
                avg_conf = np.mean(confidences)
                current_total = self.detection_stats['avg_confidence'] * self.frame_count
                self.detection_stats['avg_confidence'] = (current_total + avg_conf) / (self.frame_count + 1)
                
                # Update class counts
                for class_id in class_ids:
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    self.detection_stats['class_counts'][class_name] = \
                        self.detection_stats['class_counts'].get(class_name, 0) + 1
                        
            self.frame_count += 1
            self.detection_var.set(f"Total Detections: {self.detection_stats['total_detections']}")
            
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_var.set(f"FPS: {self.current_fps:.1f}")
            self.fps_counter = 0
            self.fps_start_time = current_time
            
    def _update_video_display(self):
        """Update the video display in GUI."""
        if self.processed_frame is not None:
            # Resize frame to fit display
            display_frame = cv2.resize(self.processed_frame, (800, 600))
            
            # Convert BGR to RGB for tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            from PIL import Image, ImageTk
            img = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
            
    def _export_statistics(self):
        """Export detection statistics to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_stats_{timestamp}.json"
        
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'total_frames_processed': self.frame_count,
            'total_detections': self.detection_stats['total_detections'],
            'average_confidence': self.detection_stats['avg_confidence'],
            'average_fps': self.current_fps,
            'class_distribution': self.detection_stats['class_counts']
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2)
            messagebox.showinfo("Export", f"Statistics exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export statistics: {str(e)}")
            
    def _save_current_frame(self):
        """Save current frame with detections."""
        if self.processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_frame_{timestamp}.jpg"
            cv2.imwrite(filename, self.processed_frame)
            messagebox.showinfo("Saved", f"Frame saved as {filename}")
        else:
            messagebox.showwarning("Save", "No frame to save")
            
    def run(self):
        """Start the application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self._stop_detection()
            if self.is_recording:
                self._stop_recording()


def main():
    """Main function to run the object detector."""
    # Check if ultralytics is installed
    if not YOLO_AVAILABLE:
        print("Installing required packages...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "pillow"])
            print("✅ Packages installed successfully!")
            print("Please restart the application.")
            return
        except Exception as e:
            print(f"❌ Failed to install packages: {e}")
            print("Please manually install: pip install ultralytics pillow")
            return
    
    print("🎯 Starting State-of-the-Art Live Object Detector...")
    print("Features:")
    print("  • YOLO11 models (nano to extra-large)")
    print("  • Real-time webcam detection")
    print("  • Video file processing")
    print("  • Advanced parameter tuning")
    print("  • Performance analytics")
    print("  • Recording capabilities")
    print("  • Export functionality")
    print()
    
    # Create and run the detector
    detector = StateOfTheArtObjectDetector()
    detector.run()


if __name__ == "__main__":
    main()
