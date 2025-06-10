#!/usr/bin/env python3
"""
AI-Enhanced Tello HUD with Ollama Integration
Combines manual controls with natural language AI commands.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import json
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import queue

# Import our controllers
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.tello_control import TelloController
from ai.ollama_controller import OllamaController


class AITelloHUD:
    """Enhanced HUD with AI natural language control."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI-Powered Tello Drone HUD")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1a1a1a')
        
        # Controllers
        self.tello_controller = None
        self.ai_controller = None
        self.connected = False
        
        # Threading
        self.sensor_thread = None
        self.video_thread = None
        self.running = False
        
        # AI Command Queue
        self.ai_command_queue = queue.Queue()
        
        # Setup UI
        self.setup_ui()
        
        # Start update loops
        self.start_update_loops()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Connection and AI status
        self.setup_top_section(main_frame)
        
        # Middle section - Split into three panels
        middle_frame = tk.Frame(main_frame, bg='#1a1a1a')
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Manual controls
        self.setup_left_panel(middle_frame)
        
        # Center panel - Video and sensors
        self.setup_center_panel(middle_frame)
        
        # Right panel - AI chat and activity
        self.setup_right_panel(middle_frame)
    
    def setup_top_section(self, parent):
        """Setup connection and status section."""
        top_frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection controls
        conn_frame = tk.Frame(top_frame, bg='#2d2d2d')
        conn_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.connect_btn = tk.Button(
            conn_frame, text="🔌 Connect Drone", 
            command=self.toggle_connection,
            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
            width=15, height=2
        )
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.ai_btn = tk.Button(
            conn_frame, text="🤖 Connect AI", 
            command=self.toggle_ai,
            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
            width=15, height=2
        )
        self.ai_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicators
        status_frame = tk.Frame(top_frame, bg='#2d2d2d')
        status_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.drone_status = tk.Label(
            status_frame, text="🔴 Drone: Disconnected",
            bg='#2d2d2d', fg='#ff4444', font=('Arial', 10, 'bold')
        )
        self.drone_status.pack(side=tk.RIGHT, padx=10)
        
        self.ai_status = tk.Label(
            status_frame, text="🔴 AI: Disconnected",
            bg='#2d2d2d', fg='#ff4444', font=('Arial', 10, 'bold')
        )
        self.ai_status.pack(side=tk.RIGHT, padx=10)
    
    def setup_left_panel(self, parent):
        """Setup manual flight controls panel."""
        left_frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.configure(width=300)
        
        # Title
        title = tk.Label(
            left_frame, text="🎮 MANUAL CONTROLS",
            bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold')
        )
        title.pack(pady=10)
        
        # Flight controls
        flight_frame = tk.Frame(left_frame, bg='#2d2d2d')
        flight_frame.pack(pady=10)
        
        # Takeoff/Land buttons
        tk.Button(
            flight_frame, text="🛫 TAKEOFF", command=self.takeoff,
            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
            width=12, height=2
        ).pack(pady=2)
        
        tk.Button(
            flight_frame, text="🛬 LAND", command=self.land,
            bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
            width=12, height=2
        ).pack(pady=2)
        
        tk.Button(
            flight_frame, text="🚨 EMERGENCY", command=self.emergency,
            bg='#F44336', fg='white', font=('Arial', 10, 'bold'),
            width=12, height=2
        ).pack(pady=2)
        
        # Movement controls
        move_frame = tk.Frame(left_frame, bg='#2d2d2d')
        move_frame.pack(pady=10)
        
        tk.Label(
            move_frame, text="Movement:", 
            bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold')
        ).pack()
        
        # Up/Down
        tk.Button(
            move_frame, text="🆙 UP", command=lambda: self.move('up'),
            bg='#9C27B0', fg='white', width=8, height=2
        ).pack(pady=1)
        
        # Forward/Back/Left/Right in cross pattern
        nav_frame = tk.Frame(move_frame, bg='#2d2d2d')
        nav_frame.pack(pady=5)
        
        tk.Button(
            nav_frame, text="⬆️🫴 FORWARD", command=lambda: self.move('forward'),
            bg='#3F51B5', fg='white', width=10, height=2
        ).pack()
        
        lr_frame = tk.Frame(nav_frame, bg='#2d2d2d')
        lr_frame.pack()
        
        tk.Button(
            lr_frame, text="⬅️ LEFT", command=lambda: self.move('left'),
            bg='#3F51B5', fg='white', width=8, height=2
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            lr_frame, text="➡️ RIGHT", command=lambda: self.move('right'),
            bg='#3F51B5', fg='white', width=8, height=2
        ).pack(side=tk.RIGHT, padx=2)
        
        tk.Button(
            nav_frame, text="🔙🤚 BACKWARD", command=lambda: self.move('back'),
            bg='#3F51B5', fg='white', width=10, height=2
        ).pack()
        
        tk.Button(
            move_frame, text="🔽 DOWN", command=lambda: self.move('down'),
            bg='#9C27B0', fg='white', width=8, height=2
        ).pack(pady=1)
        
        # Rotation controls
        rot_frame = tk.Frame(left_frame, bg='#2d2d2d')
        rot_frame.pack(pady=10)
        
        tk.Label(
            rot_frame, text="Rotation:", 
            bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold')
        ).pack()
        
        tk.Button(
            rot_frame, text="↪️ CLOCKWISE", command=lambda: self.rotate('cw'),
            bg='#FF5722', fg='white', width=12, height=2
        ).pack(pady=1)
        
        tk.Button(
            rot_frame, text="↩️ COUNTER-CW", command=lambda: self.rotate('ccw'),
            bg='#FF5722', fg='white', width=12, height=2
        ).pack(pady=1)
    
    def setup_center_panel(self, parent):
        """Setup video and sensor display panel."""
        center_frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Video display
        video_frame = tk.Frame(center_frame, bg='#000000')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(
            video_frame, text="📹 Video Feed\n(Connect drone to start)",
            bg='#000000', fg='white', font=('Arial', 14)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Sensor display
        sensor_frame = tk.Frame(center_frame, bg='#1a1a1a')
        sensor_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.sensor_text = tk.Text(
            sensor_frame, height=8, bg='#000000', fg='#00ff00',
            font=('Courier', 9), wrap=tk.WORD
        )
        self.sensor_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_right_panel(self, parent):
        """Setup AI chat and activity panel."""
        right_frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.configure(width=400)
        
        # AI Chat section
        ai_frame = tk.Frame(right_frame, bg='#2d2d2d')
        ai_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(
            ai_frame, text="🤖 AI COMMAND INTERFACE",
            bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold')
        ).pack(pady=(0, 10))
        
        # Chat history
        self.chat_history = scrolledtext.ScrolledText(
            ai_frame, height=20, bg='#1a1a1a', fg='white',
            font=('Arial', 10), wrap=tk.WORD
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Command input
        input_frame = tk.Frame(ai_frame, bg='#2d2d2d')
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.command_entry = tk.Entry(
            input_frame, bg='#333333', fg='white', font=('Arial', 10),
            insertbackground='white'
        )
        self.command_entry.pack(fill=tk.X, pady=(0, 5))
        self.command_entry.bind('<Return>', self.send_ai_command)
        
        tk.Button(
            input_frame, text="🚀 Send Command", command=self.send_ai_command,
            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')
        ).pack(fill=tk.X)
        
        # Quick commands
        quick_frame = tk.Frame(ai_frame, bg='#2d2d2d')
        quick_frame.pack(fill=tk.X)
        
        tk.Label(
            quick_frame, text="Quick Commands:",
            bg='#2d2d2d', fg='white', font=('Arial', 9, 'bold')
        ).pack(anchor=tk.W)
        
        quick_commands = [
            "Take off and hover",
            "Move forward 50 cm",
            "Turn right 45 degrees", 
            "Show sensor data",
            "Land safely"
        ]
        
        for cmd in quick_commands:
            tk.Button(
                quick_frame, text=cmd, 
                command=lambda c=cmd: self.send_quick_command(c),
                bg='#555555', fg='white', font=('Arial', 8),
                relief=tk.FLAT, bd=1
            ).pack(fill=tk.X, pady=1)
    
    def toggle_connection(self):
        """Toggle drone connection."""
        if not self.connected:
            self.connect_drone()
        else:
            self.disconnect_drone()
    
    def connect_drone(self):
        """Connect to Tello drone."""
        try:
            self.tello_controller = TelloController()
            if self.tello_controller.connect():
                self.connected = True
                self.connect_btn.configure(text="🔌 Disconnect", bg='#F44336')
                self.drone_status.configure(text="🟢 Drone: Connected", fg='#44ff44')
                self.add_chat_message("System", "✅ Drone connected successfully!")
                self.start_sensor_updates()
                self.start_video_stream()
            else:
                self.add_chat_message("System", "❌ Failed to connect to drone")
        except Exception as e:
            self.add_chat_message("System", f"❌ Connection error: {e}")
    
    def disconnect_drone(self):
        """Disconnect from drone."""
        self.running = False
        if self.tello_controller:
            self.tello_controller.disconnect()
        self.connected = False
        self.connect_btn.configure(text="🔌 Connect Drone", bg='#4CAF50')
        self.drone_status.configure(text="🔴 Drone: Disconnected", fg='#ff4444')
        self.add_chat_message("System", "🔌 Drone disconnected")
    
    def toggle_ai(self):
        """Toggle AI connection."""
        if not self.ai_controller:
            self.connect_ai()
        else:
            self.disconnect_ai()
    
    def connect_ai(self):
        """Connect to Ollama AI."""
        try:
            self.ai_controller = OllamaController()
            if self.ai_controller.check_ollama_connection():
                self.ai_controller.tello = self.tello_controller
                self.ai_btn.configure(text="🤖 Disconnect AI", bg='#F44336')
                self.ai_status.configure(text="🟢 AI: Connected", fg='#44ff44')
                self.add_chat_message("AI", "🤖 AI assistant ready! You can now use natural language commands.")
                self.add_chat_message("System", f"Using model: {self.ai_controller.model}")
            else:
                self.add_chat_message("System", "❌ Failed to connect to Ollama. Make sure it's running.")
        except Exception as e:
            self.add_chat_message("System", f"❌ AI connection error: {e}")
    
    def disconnect_ai(self):
        """Disconnect from AI."""
        self.ai_controller = None
        self.ai_btn.configure(text="🤖 Connect AI", bg='#2196F3')
        self.ai_status.configure(text="🔴 AI: Disconnected", fg='#ff4444')
        self.add_chat_message("System", "🤖 AI disconnected")
    
    def send_ai_command(self, event=None):
        """Send AI command."""
        if not self.ai_controller:
            self.add_chat_message("System", "❌ AI not connected")
            return
        
        command = self.command_entry.get().strip()
        if not command:
            return
        
        self.command_entry.delete(0, tk.END)
        self.add_chat_message("You", command)
        
        # Process AI command in background thread
        threading.Thread(target=self.process_ai_command, args=(command,), daemon=True).start()
    
    def send_quick_command(self, command):
        """Send a quick command."""
        if not self.ai_controller:
            self.add_chat_message("System", "❌ AI not connected")
            return
        
        self.add_chat_message("You", command)
        threading.Thread(target=self.process_ai_command, args=(command,), daemon=True).start()
    
    def process_ai_command(self, command):
        """Process AI command in background."""
        try:
            self.add_chat_message("AI", "🤔 Processing command...")
            success = self.ai_controller.process_natural_language_command(command)
            
            if success:
                self.add_chat_message("AI", "✅ Command executed successfully!")
            else:
                self.add_chat_message("AI", "❌ Command failed or was blocked for safety")
        except Exception as e:
            self.add_chat_message("AI", f"❌ Error processing command: {e}")
    
    def add_chat_message(self, sender, message):
        """Add message to chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding
        colors = {
            "You": "#4CAF50",
            "AI": "#2196F3", 
            "System": "#FF9800"
        }
        
        self.chat_history.configure(state=tk.NORMAL)
        
        # Add timestamp and sender
        self.chat_history.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        self.chat_history.insert(tk.END, f"{sender}: ", ('sender', sender))
        self.chat_history.insert(tk.END, f"{message}\n\n")
        
        # Configure tags for colors
        self.chat_history.tag_configure('timestamp', foreground='#888888')
        for name, color in colors.items():
            self.chat_history.tag_configure(('sender', name), foreground=color, font=('Arial', 10, 'bold'))
        
        self.chat_history.configure(state=tk.DISABLED)
        self.chat_history.see(tk.END)
    
    # Manual control methods
    def takeoff(self):
        """Manual takeoff."""
        if self.tello_controller and self.connected:
            success = self.tello_controller.takeoff()
            self.add_chat_message("System", "🛫 Takeoff" + (" successful" if success else " failed"))
    
    def land(self):
        """Manual landing."""
        if self.tello_controller and self.connected:
            success = self.tello_controller.land()
            self.add_chat_message("System", "🛬 Landing" + (" successful" if success else " failed"))
    
    def emergency(self):
        """Emergency stop."""
        if self.tello_controller and self.connected:
            self.tello_controller.emergency_stop()
            self.add_chat_message("System", "🚨 Emergency stop activated")
    
    def move(self, direction):
        """Move in specified direction."""
        if not self.tello_controller or not self.connected:
            return
        
        distance = 30  # Default distance
        
        if direction == 'up':
            self.tello_controller.move_up(distance)
        elif direction == 'down':
            self.tello_controller.move_down(distance)
        elif direction == 'forward':
            self.tello_controller.move_forward(distance)
        elif direction == 'back':
            self.tello_controller.move_back(distance)
        elif direction == 'left':
            self.tello_controller.move_left(distance)
        elif direction == 'right':
            self.tello_controller.move_right(distance)
        
        self.add_chat_message("System", f"📍 Moving {direction} {distance}cm")
    
    def rotate(self, direction):
        """Rotate in specified direction."""
        if not self.tello_controller or not self.connected:
            return
        
        degrees = 30  # Default rotation
        
        if direction == 'cw':
            self.tello_controller.rotate_clockwise(degrees)
        elif direction == 'ccw':
            self.tello_controller.rotate_counter_clockwise(degrees)
        
        self.add_chat_message("System", f"🔄 Rotating {direction} {degrees}°")
    
    def start_sensor_updates(self):
        """Start sensor update thread."""
        self.running = True
        self.sensor_thread = threading.Thread(target=self.sensor_update_loop, daemon=True)
        self.sensor_thread.start()
    
    def sensor_update_loop(self):
        """Sensor update loop."""
        while self.running and self.connected:
            try:
                if self.tello_controller:
                    sensors = self.tello_controller.get_all_sensors()
                    self.update_sensor_display(sensors)
                time.sleep(1.0)  # Update every second
            except Exception as e:
                print(f"Sensor update error: {e}")
    
    def update_sensor_display(self, sensors):
        """Update sensor display."""
        if not sensors:
            return
        
        display_text = f"""
🔋 Battery: {sensors.get('battery_percent', 0)}%
📏 Height: {sensors.get('height_cm', 0)} cm
🌡️ Temperature: {sensors.get('temperature_avg_celsius', 0):.1f}°C
🎯 Attitude: P:{sensors.get('pitch_degrees', 0)}° R:{sensors.get('roll_degrees', 0)}° Y:{sensors.get('yaw_degrees', 0)}°
🏃 Speed: {sensors.get('velocity_total_cms', 0):.1f} cm/s
⚡ Accel: {sensors.get('acceleration_total_mg', 0):.3f} mg
📡 WiFi: {sensors.get('wifi_signal_noise_ratio', 'N/A')}
        """
        
        self.root.after(0, lambda: self.sensor_text.delete(1.0, tk.END))
        self.root.after(0, lambda: self.sensor_text.insert(1.0, display_text.strip()))
    
    def start_video_stream(self):
        """Start video stream."""
        if self.tello_controller:
            self.tello_controller.start_video()
            self.video_thread = threading.Thread(target=self.video_update_loop, daemon=True)
            self.video_thread.start()
    
    def video_update_loop(self):
        """Video update loop."""
        while self.running and self.connected:
            try:
                if self.tello_controller:
                    frame = self.tello_controller.get_frame()
                    if frame is not None:
                        self.update_video_display(frame)
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Video update error: {e}")
    
    def update_video_display(self, frame):
        """Update video display."""
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            max_width, max_height = 640, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            # Update display
            self.root.after(0, lambda: self.video_label.configure(image=photo, text=""))
            self.root.after(0, lambda: setattr(self.video_label, 'image', photo))
            
        except Exception as e:
            print(f"Video display error: {e}")
    
    def start_update_loops(self):
        """Start periodic update loops."""
        # This method can be used for additional periodic updates if needed
        pass
    
    def on_closing(self):
        """Handle window closing."""
        self.running = False
        if self.connected:
            self.disconnect_drone()
        self.root.destroy()


def main():
    """Main function."""
    root = tk.Tk()
    app = AITelloHUD(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main() 