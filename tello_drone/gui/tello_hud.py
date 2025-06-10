"""
Tello Drone Interactive HUD (Heads-Up Display)
==============================================

Modern, interactive HUD for real-time Tello drone sensor monitoring and control.
Features attitude indicators, sensor gauges, flight controls, and data logging.

Requirements:
    pip install djitellopy tkinter matplotlib numpy

Usage:
    python tello_hud.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import math
from datetime import datetime
from typing import Dict, Any
import numpy as np

# Import the Tello controller
from ..core.tello_control import TelloController

class TelloHUD:
    """Interactive HUD for Tello drone monitoring and control."""
    
    def __init__(self):
        self.controller = TelloController()
        self.running = False
        self.update_thread = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("🚁 Tello Drone HUD - Interactive Control & Monitoring")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Color scheme
        self.colors = {
            'bg_primary': '#1a1a1a',
            'bg_secondary': '#2d2d2d',
            'bg_panel': '#3a3a3a',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'accent_green': '#00ff88',
            'accent_blue': '#00aaff',
            'accent_red': '#ff4444',
            'accent_yellow': '#ffaa00',
            'accent_purple': '#aa44ff'
        }
        
        # Current sensor data
        self.sensor_data = {}
        self.data_count = 0
        
        # Video streaming
        self.video_streaming = False
        self.video_thread = None
        
        # Create the interface
        self.create_interface()
        
        # Configure styles
        self.configure_styles()
        
    def configure_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme
        style.configure('Dark.TFrame', background=self.colors['bg_secondary'])
        style.configure('Panel.TFrame', background=self.colors['bg_panel'])
        style.configure('Dark.TLabel', background=self.colors['bg_secondary'], 
                       foreground=self.colors['text_primary'], font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=self.colors['bg_secondary'], 
                       foreground=self.colors['accent_blue'], font=('Segoe UI', 12, 'bold'))
        style.configure('Value.TLabel', background=self.colors['bg_panel'], 
                       foreground=self.colors['accent_green'], font=('Consolas', 11, 'bold'))
        style.configure('Status.TLabel', background=self.colors['bg_panel'], 
                       foreground=self.colors['text_primary'], font=('Segoe UI', 9))
        
    def create_interface(self):
        """Create the main HUD interface."""
        
        # Top control bar
        self.create_control_bar()
        
        # Main content area
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Flight status and controls
        left_panel = ttk.Frame(main_frame, style='Panel.TFrame')
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        self.create_flight_panel(left_panel)
        
        # Center panel - Sensor displays
        center_panel = ttk.Frame(main_frame, style='Dark.TFrame')
        center_panel.pack(side='left', fill='both', expand=True, padx=5)
        self.create_sensor_panels(center_panel)
        
        # Right panel - Attitude indicator only
        right_panel = ttk.Frame(main_frame, style='Panel.TFrame')
        right_panel.pack(side='right', fill='y', padx=(5, 0))
        self.create_attitude_panel(right_panel)
        
        # Bottom status bar
        self.create_status_bar()
        
    def create_control_bar(self):
        """Create the top control bar."""
        control_frame = ttk.Frame(self.root, style='Panel.TFrame')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Connection controls
        conn_frame = ttk.Frame(control_frame, style='Panel.TFrame')
        conn_frame.pack(side='left')
        
        self.connect_btn = tk.Button(conn_frame, text="🔌 Connect", 
                                   command=self.connect_drone,
                                   bg=self.colors['accent_blue'], fg='white',
                                   font=('Segoe UI', 10, 'bold'), width=10)
        self.connect_btn.pack(side='left', padx=2)
        
        self.disconnect_btn = tk.Button(conn_frame, text="❌ Disconnect", 
                                      command=self.disconnect_drone,
                                      bg=self.colors['accent_red'], fg='white',
                                      font=('Segoe UI', 10, 'bold'), width=10)
        self.disconnect_btn.pack(side='left', padx=2)
        
        # Monitoring controls
        monitor_frame = ttk.Frame(control_frame, style='Panel.TFrame')
        monitor_frame.pack(side='left', padx=20)
        
        self.start_btn = tk.Button(monitor_frame, text="▶️ Start HUD", 
                                 command=self.start_monitoring,
                                 bg=self.colors['accent_green'], fg='white',
                                 font=('Segoe UI', 10, 'bold'), width=10)
        self.start_btn.pack(side='left', padx=2)
        
        self.stop_btn = tk.Button(monitor_frame, text="⏹️ Stop HUD", 
                                command=self.stop_monitoring,
                                bg=self.colors['accent_yellow'], fg='white',
                                font=('Segoe UI', 10, 'bold'), width=10)
        self.stop_btn.pack(side='left', padx=2)
        
        # Data controls
        data_frame = ttk.Frame(control_frame, style='Panel.TFrame')
        data_frame.pack(side='right')
        
        self.save_btn = tk.Button(data_frame, text="💾 Save Data", 
                                command=self.save_data,
                                bg=self.colors['accent_purple'], fg='white',
                                font=('Segoe UI', 10, 'bold'), width=10)
        self.save_btn.pack(side='left', padx=2)
        
        # Video controls
        video_frame = ttk.Frame(control_frame, style='Panel.TFrame')
        video_frame.pack(side='right', padx=(20, 0))
        
        self.video_btn = tk.Button(video_frame, text="📹 Start Video", 
                                 command=self.toggle_video,
                                 bg=self.colors['bg_secondary'], fg='white',
                                 font=('Segoe UI', 10, 'bold'), width=12)
        self.video_btn.pack(side='left', padx=2)
        
    def create_flight_panel(self, parent):
        """Create the flight status and control panel."""
        # Title
        title = ttk.Label(parent, text="🚁 FLIGHT CONTROL", style='Title.TLabel')
        title.pack(pady=(10, 5))
        
        # Connection status
        status_frame = ttk.Frame(parent, style='Panel.TFrame')
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.connection_status = ttk.Label(status_frame, text="🔴 DISCONNECTED", 
                                         style='Status.TLabel')
        self.connection_status.pack()
        
        self.flight_status = ttk.Label(status_frame, text="🛬 LANDED", 
                                     style='Status.TLabel')
        self.flight_status.pack()
        
        # Flight controls
        controls_frame = ttk.Frame(parent, style='Panel.TFrame')
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Basic flight controls
        basic_label = ttk.Label(controls_frame, text="Basic Controls:", style='Dark.TLabel')
        basic_label.pack()
        
        btn_frame1 = ttk.Frame(controls_frame, style='Panel.TFrame')
        btn_frame1.pack(pady=2)
        
        self.takeoff_btn = tk.Button(btn_frame1, text="🛫 Takeoff", 
                                   command=self.takeoff_drone,
                                   bg=self.colors['accent_green'], fg='white',
                                   font=('Segoe UI', 9), width=8)
        self.takeoff_btn.pack(side='left', padx=1)
        
        self.land_btn = tk.Button(btn_frame1, text="🛬 Land", 
                                command=self.land_drone,
                                bg=self.colors['accent_blue'], fg='white',
                                font=('Segoe UI', 9), width=8)
        self.land_btn.pack(side='left', padx=1)
        
        # Emergency stop
        emergency_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        emergency_frame.pack(pady=5)
        
        self.emergency_btn = tk.Button(emergency_frame, text="🚨 EMERGENCY", 
                                     command=self.emergency_stop,
                                     bg=self.colors['accent_red'], fg='white',
                                     font=('Segoe UI', 10, 'bold'), width=16)
        self.emergency_btn.pack()
        
        # Movement controls
        move_label = ttk.Label(controls_frame, text="Movement (20cm):", style='Dark.TLabel')
        move_label.pack(pady=(10, 0))
        
        # Up/Down
        ud_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        ud_frame.pack(pady=2)
        
        tk.Button(ud_frame, text="🆙 \n UP", command=lambda: self.move_drone('up'),
                 bg=self.colors['bg_secondary'], fg='white', width=8, height=2).pack(side='left', padx=1)
        tk.Button(ud_frame, text="🔽 \n DOWN ", command=lambda: self.move_drone('down'),
                 bg=self.colors['bg_secondary'], fg='white', width=8, height=2).pack(side='left', padx=1)
        
        # Forward/Back
        fb_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        fb_frame.pack(pady=2)
        
        tk.Button(fb_frame, text="⬆️🫴 \n FORWARD", command=lambda: self.move_drone('forward'),
                 bg=self.colors['bg_secondary'], fg='white', width=10, height=2).pack(side='left', padx=1)
        tk.Button(fb_frame, text="🔙🤚 \n BACKWARD", command=lambda: self.move_drone('back'),
                 bg=self.colors['bg_secondary'], fg='white', width=10, height=2).pack(side='left', padx=1)
        
        # Left/Right
        lr_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        lr_frame.pack(pady=2)
        
        tk.Button(lr_frame, text="⬅️ \n LEFT", command=lambda: self.move_drone('left'),
                 bg=self.colors['bg_secondary'], fg='white', width=8, height=2).pack(side='left', padx=1)
        tk.Button(lr_frame, text="➡️ \n RIGHT", command=lambda: self.move_drone('right'),
                 bg=self.colors['bg_secondary'], fg='white', width=8, height=2).pack(side='left', padx=1)
        
        # Rotation controls
        rot_label = ttk.Label(controls_frame, text="Rotation (15°):", style='Dark.TLabel')
        rot_label.pack(pady=(10, 0))
        
        rot_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        rot_frame.pack(pady=2)
        
        tk.Button(rot_frame, text="↪️ \n CLOCKWISE", command=lambda: self.rotate_drone('cw'),
                 bg=self.colors['bg_secondary'], fg='white', width=12, height=2).pack(side='left', padx=1)
        tk.Button(rot_frame, text="↩️ \n COUNTER-CW", command=lambda: self.rotate_drone('ccw'),
                 bg=self.colors['bg_secondary'], fg='white', width=12, height=2).pack(side='left', padx=1)
        
    def create_sensor_panels(self, parent):
        """Create the main sensor display panels."""
        
        # Top row - Power and environment
        top_frame = ttk.Frame(parent, style='Dark.TFrame')
        top_frame.pack(fill='x', pady=(0, 5))
        
        # Battery panel
        self.create_battery_panel(top_frame)
        
        # Environment panel
        self.create_environment_panel(top_frame)
        
        # Middle row - Motion sensors
        middle_frame = ttk.Frame(parent, style='Dark.TFrame')
        middle_frame.pack(fill='x', pady=5)
        
        # Velocity panel
        self.create_velocity_panel(middle_frame)
        
        # Acceleration panel
        self.create_acceleration_panel(middle_frame)
        
        # Bottom row - System info
        bottom_frame = ttk.Frame(parent, style='Dark.TFrame')
        bottom_frame.pack(fill='x', pady=(5, 0))
        
        # System info panel
        self.create_system_panel(bottom_frame)
        
        # Video display panel
        video_frame = ttk.Frame(parent, style='Dark.TFrame')
        video_frame.pack(fill='x', pady=(5, 0))
        self.create_video_panel(video_frame)
        
    def create_battery_panel(self, parent):
        """Create battery and power status panel."""
        panel = ttk.Frame(parent, style='Panel.TFrame')
        panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        title = ttk.Label(panel, text="🔋 POWER STATUS", style='Title.TLabel')
        title.pack(pady=5)
        
        # Battery percentage with visual bar
        self.battery_frame = ttk.Frame(panel, style='Panel.TFrame')
        self.battery_frame.pack(pady=5)
        
        self.battery_label = ttk.Label(self.battery_frame, text="Battery: ---%", style='Value.TLabel')
        self.battery_label.pack()
        
        # Create battery bar canvas
        self.battery_canvas = tk.Canvas(self.battery_frame, width=200, height=30, 
                                      bg=self.colors['bg_secondary'], highlightthickness=0)
        self.battery_canvas.pack(pady=5)
        
        # Flight time
        self.flight_time_label = ttk.Label(panel, text="Flight Time: ---s", style='Value.TLabel')
        self.flight_time_label.pack()
        
        # Height
        self.height_label = ttk.Label(panel, text="Height: ---cm", style='Value.TLabel')
        self.height_label.pack()
        
    def create_environment_panel(self, parent):
        """Create environment sensors panel."""
        panel = ttk.Frame(parent, style='Panel.TFrame')
        panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        title = ttk.Label(panel, text="🌡️ ENVIRONMENT", style='Title.TLabel')
        title.pack(pady=5)
        
        # Temperature
        self.temp_label = ttk.Label(panel, text="Temperature: ---°C", style='Value.TLabel')
        self.temp_label.pack(pady=2)
        
        self.temp_range_label = ttk.Label(panel, text="Range: ---°C - ---°C", style='Status.TLabel')
        self.temp_range_label.pack()
        
        # Barometer (altitude measurement, not pressure)
        self.baro_label = ttk.Label(panel, text="Baro Altitude: --- cm", style='Value.TLabel')
        self.baro_label.pack(pady=2)
        
        # WiFi signal
        self.wifi_label = ttk.Label(panel, text="WiFi SNR: ---", style='Value.TLabel')
        self.wifi_label.pack(pady=2)
        
        # Distance sensor
        self.distance_label = ttk.Label(panel, text="ToF Distance: ---cm", style='Value.TLabel')
        self.distance_label.pack(pady=2)
        
    def create_velocity_panel(self, parent):
        """Create velocity display panel."""
        panel = ttk.Frame(parent, style='Panel.TFrame')
        panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        title = ttk.Label(panel, text="🏃 VELOCITY (cm/s)", style='Title.TLabel')
        title.pack(pady=5)
        
        # Velocity components
        self.vel_x_label = ttk.Label(panel, text="X (L/R): +---.-- cm/s", style='Value.TLabel')
        self.vel_x_label.pack()
        
        self.vel_y_label = ttk.Label(panel, text="Y (F/B): +---.-- cm/s", style='Value.TLabel')
        self.vel_y_label.pack()
        
        self.vel_z_label = ttk.Label(panel, text="Z (U/D): +---.-- cm/s", style='Value.TLabel')
        self.vel_z_label.pack()
        
        self.vel_total_label = ttk.Label(panel, text="Total: ---.-- cm/s", style='Value.TLabel')
        self.vel_total_label.pack(pady=(5, 0))
        
    def create_acceleration_panel(self, parent):
        """Create acceleration display panel."""
        panel = ttk.Frame(parent, style='Panel.TFrame')
        panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        title = ttk.Label(panel, text="⚡ ACCELERATION (milli-g)", style='Title.TLabel')
        title.pack(pady=5)
        
        # Acceleration components
        self.accel_x_label = ttk.Label(panel, text="X (L/R): +---.--- mg", style='Value.TLabel')
        self.accel_x_label.pack()
        
        self.accel_y_label = ttk.Label(panel, text="Y (F/B): +---.--- mg", style='Value.TLabel')
        self.accel_y_label.pack()
        
        self.accel_z_label = ttk.Label(panel, text="Z (U/D): +---.--- mg", style='Value.TLabel')
        self.accel_z_label.pack()
        
        self.accel_total_label = ttk.Label(panel, text="Total: ---.--- mg", style='Value.TLabel')
        self.accel_total_label.pack(pady=(5, 0))
        
    def create_system_panel(self, parent):
        """Create system information panel."""
        panel = ttk.Frame(parent, style='Panel.TFrame')
        panel.pack(fill='x')
        
        title = ttk.Label(panel, text="📊 SYSTEM INFO", style='Title.TLabel')
        title.pack(pady=5)
        
        info_frame = ttk.Frame(panel, style='Panel.TFrame')
        info_frame.pack()
        
        # Last update time
        self.update_time_label = ttk.Label(info_frame, text="Last Update: ---", style='Status.TLabel')
        self.update_time_label.pack(side='left', padx=10)
        
        # Video status
        self.video_status_label = ttk.Label(info_frame, text="📹 Video: OFF", style='Status.TLabel')
        self.video_status_label.pack(side='left', padx=10)
        
        # Update rate
        self.update_rate_label = ttk.Label(info_frame, text="Update Rate: --- Hz", style='Status.TLabel')
        self.update_rate_label.pack(side='left', padx=10)
        
    def create_video_panel(self, parent):
        """Create video display panel."""
        panel = ttk.Frame(parent, style='Panel.TFrame')
        panel.pack(fill='x')
        
        title = ttk.Label(panel, text="📹 VIDEO STREAM", style='Title.TLabel')
        title.pack(pady=5)
        
        # Video display frame - larger since it's now in the center
        self.video_frame = tk.Frame(panel, bg=self.colors['bg_secondary'], 
                                   width=640, height=480)
        self.video_frame.pack(pady=5)
        self.video_frame.pack_propagate(False)
        
        # Video canvas
        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480,
                                    bg=self.colors['bg_secondary'], highlightthickness=0)
        self.video_canvas.pack()
        
        # Initial video display
        self.video_canvas.create_text(320, 240, text="📹 Video Stream Inactive\n\nConnect to Tello and click\n'Start Video' to begin streaming", 
                                    fill='white', font=('Arial', 14), justify='center')
        
        # Video status
        self.video_status_text = ttk.Label(panel, text="Video: OFF", style='Status.TLabel')
        self.video_status_text.pack()

    def create_attitude_panel(self, parent):
        """Create attitude indicator panel."""
        title = ttk.Label(parent, text="🎯 ATTITUDE", style='Title.TLabel')
        title.pack(pady=(10, 5))
        
        # Attitude indicator canvas
        self.attitude_canvas = tk.Canvas(parent, width=250, height=250, 
                                       bg=self.colors['bg_secondary'], highlightthickness=0)
        self.attitude_canvas.pack(pady=10)
        
        # Attitude values
        attitude_frame = ttk.Frame(parent, style='Panel.TFrame')
        attitude_frame.pack(pady=5)
        
        self.pitch_label = ttk.Label(attitude_frame, text="Pitch: ---°", style='Value.TLabel')
        self.pitch_label.pack()
        
        self.roll_label = ttk.Label(attitude_frame, text="Roll: ---°", style='Value.TLabel')
        self.roll_label.pack()
        
        self.yaw_label = ttk.Label(attitude_frame, text="Yaw: ---°", style='Value.TLabel')
        self.yaw_label.pack()
        
        # Activity log
        log_title = ttk.Label(parent, text="📜 ACTIVITY LOG", style='Title.TLabel')
        log_title.pack(pady=(20, 5))
        
        # Create scrollable text widget for logs
        log_frame = ttk.Frame(parent, style='Panel.TFrame')
        log_frame.pack(fill='both', expand=True, padx=10)
        
        self.log_text = tk.Text(log_frame, height=8, width=30, 
                              bg=self.colors['bg_secondary'], 
                              fg=self.colors['text_secondary'],
                              font=('Consolas', 8))
        log_scroll = tk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')
        
    def create_status_bar(self):
        """Create bottom status bar."""
        status_frame = ttk.Frame(self.root, style='Panel.TFrame')
        status_frame.pack(fill='x', side='bottom', padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready - Connect to Tello to begin", 
                                    style='Status.TLabel')
        self.status_label.pack(side='left')
        
        # Data counter
        self.data_counter_label = ttk.Label(status_frame, text="Data Points: 0", 
                                          style='Status.TLabel')
        self.data_counter_label.pack(side='right')
        
    def draw_battery_bar(self, percentage):
        """Draw battery percentage bar."""
        self.battery_canvas.delete("all")
        
        # Draw battery outline
        self.battery_canvas.create_rectangle(10, 8, 180, 22, outline='white', width=2)
        self.battery_canvas.create_rectangle(180, 12, 185, 18, fill='white')
        
        # Color based on battery level
        if percentage > 50:
            color = self.colors['accent_green']
        elif percentage > 20:
            color = self.colors['accent_yellow']
        else:
            color = self.colors['accent_red']
        
        # Fill battery bar
        fill_width = int((percentage / 100) * 168)
        if fill_width > 0:
            self.battery_canvas.create_rectangle(12, 10, 12 + fill_width, 20, 
                                               fill=color, outline="")
        
        # Add percentage text
        self.battery_canvas.create_text(95, 15, text=f"{percentage}%", 
                                      fill='white', font=('Segoe UI', 10, 'bold'))
        
    def draw_attitude_indicator(self, pitch, roll, yaw):
        """Draw artificial horizon attitude indicator with proper roll rotation."""
        self.attitude_canvas.delete("all")
        
        canvas_size = 250
        center = canvas_size // 2
        radius = 100
        
        # Convert angles to radians
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(-roll)  # Negative for correct rotation direction
        
        # Pitch offset (vertical displacement of horizon)
        pitch_offset = pitch * 1.5  # Scale factor for pitch sensitivity
        
        # Create clipping circle for the attitude indicator
        clip_radius = radius - 5
        
        # Draw background (create large rectangles that will be clipped)
        large_size = canvas_size * 2
        
        # Sky (blue) - extends far up from horizon line
        sky_points = []
        ground_points = []
        
        # Create horizon line points accounting for roll
        horizon_length = large_size
        for x in range(-horizon_length//2, horizon_length//2, 10):
            # Rotate the horizon line by roll angle
            rotated_x = x * math.cos(roll_rad) + center
            rotated_y = x * math.sin(roll_rad) + center + pitch_offset
            
            # Sky points (above horizon)
            sky_points.extend([
                rotated_x, rotated_y - large_size,
                rotated_x, rotated_y
            ])
            
            # Ground points (below horizon)
            ground_points.extend([
                rotated_x, rotated_y,
                rotated_x, rotated_y + large_size
            ])
        
        # Draw sky and ground as polygons
        if len(sky_points) >= 6:
            # Sky polygon
            sky_poly = [
                center - large_size, center - large_size,
                center + large_size, center - large_size,
                center + large_size, center + pitch_offset,
                center - large_size, center + pitch_offset
            ]
            self.attitude_canvas.create_polygon(sky_poly, fill='#4da6ff', outline="")
            
            # Ground polygon  
            ground_poly = [
                center - large_size, center + pitch_offset,
                center + large_size, center + pitch_offset,
                center + large_size, center + large_size,
                center - large_size, center + large_size
            ]
            self.attitude_canvas.create_polygon(ground_poly, fill='#8b4513', outline="")
        
        # Draw rotated horizon line
        horizon_half_length = 80
        h_x1 = center - horizon_half_length * math.cos(roll_rad)
        h_y1 = center + pitch_offset - horizon_half_length * math.sin(roll_rad)
        h_x2 = center + horizon_half_length * math.cos(roll_rad)
        h_y2 = center + pitch_offset + horizon_half_length * math.sin(roll_rad)
        
        self.attitude_canvas.create_line(h_x1, h_y1, h_x2, h_y2, 
                                       fill='white', width=3)
        
        # Draw pitch ladder (rotated lines)
        for p in range(-30, 31, 10):
            if p == 0:
                continue
            # Calculate position for this pitch line
            pitch_y_offset = -p * 2  # Negative because screen Y increases downward
            line_length = 30 if p % 20 == 0 else 20
            
            # Center of this pitch line
            line_center_y = center + pitch_offset + pitch_y_offset
            
            # Rotate the pitch line by roll angle
            lx1 = center - line_length * math.cos(roll_rad)
            ly1 = line_center_y - line_length * math.sin(roll_rad)
            lx2 = center + line_length * math.cos(roll_rad)
            ly2 = line_center_y + line_length * math.sin(roll_rad)
            
            # Only draw if within visible area
            if (abs(lx1 - center) < radius and abs(ly1 - center) < radius and
                abs(lx2 - center) < radius and abs(ly2 - center) < radius):
                
                self.attitude_canvas.create_line(lx1, ly1, lx2, ly2,
                                               fill='white', width=1)
                
                # Add pitch angle text
                text_x = center + (line_length + 10) * math.cos(roll_rad)
                text_y = line_center_y + (line_length + 10) * math.sin(roll_rad)
                
                if abs(text_x - center) < radius and abs(text_y - center) < radius:
                    self.attitude_canvas.create_text(text_x, text_y,
                                                   text=f"{p}°", fill='white',
                                                   font=('Arial', 7))
        
        # Draw aircraft symbol (always centered, doesn't rotate)
        # Aircraft center line (wings)
        wing_length = 25
        self.attitude_canvas.create_line(center - wing_length, center,
                                       center + wing_length, center,
                                       fill='yellow', width=4)
        
        # Aircraft center dot
        self.attitude_canvas.create_oval(center - 3, center - 3,
                                       center + 3, center + 3,
                                       fill='yellow', outline='orange', width=2)
        
        # Aircraft nose indicator (small line pointing up)
        self.attitude_canvas.create_line(center, center - 8,
                                       center, center - 15,
                                       fill='yellow', width=3)
        
        # Draw roll angle markings around the edge
        for angle in range(0, 360, 30):
            angle_rad = math.radians(angle - 90)  # Start from top
            x1 = center + (radius - 15) * math.cos(angle_rad)
            y1 = center + (radius - 15) * math.sin(angle_rad)
            x2 = center + radius * math.cos(angle_rad)
            y2 = center + radius * math.sin(angle_rad)
            
            if angle in [0, 90, 180, 270]:
                self.attitude_canvas.create_line(x1, y1, x2, y2, fill='white', width=2)
            else:
                self.attitude_canvas.create_line(x1, y1, x2, y2, fill='white', width=1)
        
        # Draw roll indicator triangle
        roll_indicator_radius = radius + 10
        roll_angle_rad = math.radians(-roll - 90)  # Point to current roll
        tri_x = center + roll_indicator_radius * math.cos(roll_angle_rad)
        tri_y = center + roll_indicator_radius * math.sin(roll_angle_rad)
        
        # Triangle points
        tri_size = 8
        tri_points = [
            tri_x, tri_y - tri_size,
            tri_x - tri_size//2, tri_y + tri_size//2,
            tri_x + tri_size//2, tri_y + tri_size//2
        ]
        self.attitude_canvas.create_polygon(tri_points, fill='red', outline='white')
        
        # Draw outer ring
        self.attitude_canvas.create_oval(center - radius, center - radius,
                                       center + radius, center + radius,
                                       outline='white', width=2)
        
        # Yaw indicator
        yaw_text = f"YAW: {yaw:+4.0f}°"
        self.attitude_canvas.create_text(center, 20, text=yaw_text,
                                       fill='white', font=('Arial', 10, 'bold'))
        
    def log_message(self, message):
        """Add message to activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Keep log to reasonable size
        lines = self.log_text.index('end-1c').split('.')[0]
        if int(lines) > 100:
            self.log_text.delete('1.0', '20.0')
    
    def update_sensor_displays(self):
        """Update all sensor displays with current data."""
        if not self.sensor_data:
            return
        
        try:
            # Update battery
            battery = self.sensor_data.get('battery_percent', 0)
            self.battery_label.config(text=f"Battery: {battery}%")
            self.draw_battery_bar(battery)
            
            # Update flight time and height
            flight_time = self.sensor_data.get('flight_time_seconds', 0)
            height = self.sensor_data.get('height_cm', 0)
            self.flight_time_label.config(text=f"Flight Time: {flight_time}s")
            self.height_label.config(text=f"Height: {height}cm")
            
            # Update environment
            temp = self.sensor_data.get('temperature_avg_celsius', 0)
            temp_high = self.sensor_data.get('temperature_high_celsius', 0)
            temp_low = self.sensor_data.get('temperature_low_celsius', 0)
            baro = self.sensor_data.get('barometer_altitude_cm', 0)
            wifi = self.sensor_data.get('wifi_signal_noise_ratio', 'N/A')
            distance = self.sensor_data.get('distance_tof_cm', 0)
            
            self.temp_label.config(text=f"Temperature: {temp:.1f}°C")
            self.temp_range_label.config(text=f"Range: {temp_low:.1f}°C - {temp_high:.1f}°C")
            self.baro_label.config(text=f"Baro Altitude: {baro:.1f} cm")
            self.wifi_label.config(text=f"WiFi SNR: {wifi}")
            self.distance_label.config(text=f"ToF Distance: {distance}cm")
            
            # Update velocity
            vel_x = self.sensor_data.get('velocity_x_cms', 0)
            vel_y = self.sensor_data.get('velocity_y_cms', 0)
            vel_z = self.sensor_data.get('velocity_z_cms', 0)
            vel_total = self.sensor_data.get('velocity_total_cms', 0)
            
            self.vel_x_label.config(text=f"X (L/R): {vel_x:+7.1f} cm/s")
            self.vel_y_label.config(text=f"Y (F/B): {vel_y:+7.1f} cm/s")
            self.vel_z_label.config(text=f"Z (U/D): {vel_z:+7.1f} cm/s")
            self.vel_total_label.config(text=f"Total: {vel_total:7.1f} cm/s")
            
            # Update acceleration (in milli-g units)
            accel_x = self.sensor_data.get('acceleration_x_mg', 0)
            accel_y = self.sensor_data.get('acceleration_y_mg', 0)
            accel_z = self.sensor_data.get('acceleration_z_mg', 0)
            accel_total = self.sensor_data.get('acceleration_total_mg', 0)
            
            self.accel_x_label.config(text=f"X (L/R): {accel_x:+8.3f} mg")
            self.accel_y_label.config(text=f"Y (F/B): {accel_y:+8.3f} mg")
            self.accel_z_label.config(text=f"Z (U/D): {accel_z:+8.3f} mg")
            self.accel_total_label.config(text=f"Total: {accel_total:8.3f} mg")
            
            # Update attitude
            pitch = self.sensor_data.get('pitch_degrees', 0)
            roll = self.sensor_data.get('roll_degrees', 0)
            yaw = self.sensor_data.get('yaw_degrees', 0)
            
            self.pitch_label.config(text=f"Pitch: {pitch:+4.0f}°")
            self.roll_label.config(text=f"Roll: {roll:+4.0f}°")
            self.yaw_label.config(text=f"Yaw: {yaw:+4.0f}°")
            
            self.draw_attitude_indicator(pitch, roll, yaw)
            
            # Update system info
            update_time = self.sensor_data.get('datetime', 'Unknown')
            video_status = "ON" if self.sensor_data.get('video_streaming', False) else "OFF"
            
            self.update_time_label.config(text=f"Last Update: {update_time[-12:]}")
            self.video_status_label.config(text=f"📹 Video: {video_status}")
            
            # Update connection and flight status
            if self.sensor_data.get('is_connected', False):
                self.connection_status.config(text="🟢 CONNECTED")
            else:
                self.connection_status.config(text="🔴 DISCONNECTED")
                
            if self.sensor_data.get('is_flying', False):
                self.flight_status.config(text="🛫 FLYING")
            else:
                self.flight_status.config(text="🛬 LANDED")
                
        except Exception as e:
            self.log_message(f"Display update error: {e}")
    
    def data_update_loop(self):
        """Main data update loop."""
        update_count = 0
        last_time = time.time()
        
        while self.running:
            try:
                # Get sensor data
                self.sensor_data = self.controller.get_all_sensors()
                
                if self.sensor_data:
                    self.data_count += 1
                    update_count += 1
                    
                    # Calculate update rate
                    current_time = time.time()
                    if current_time - last_time >= 1.0:
                        rate = update_count / (current_time - last_time)
                        self.root.after(0, lambda: self.update_rate_label.config(
                            text=f"Update Rate: {rate:.1f} Hz"))
                        self.root.after(0, lambda: self.data_counter_label.config(
                            text=f"Data Points: {self.data_count}"))
                        last_time = current_time
                        update_count = 0
                    
                    # Update displays on main thread
                    self.root.after(0, self.update_sensor_displays)
                
                time.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"Data loop error: {e}"))
                time.sleep(1)
    
    def video_update_loop(self):
        """Video streaming loop."""
        try:
            import cv2
            from PIL import Image, ImageTk
        except ImportError:
            self.log_message("❌ OpenCV and PIL required for video streaming")
            self.log_message("💡 Install with: pip install opencv-python pillow")
            return
        
        # Wait for video stream to stabilize
        self.root.after(0, lambda: self.log_message("📹 Waiting for video stream to initialize..."))
        time.sleep(3)  # Give video more time to start
        
        frame_count = 0
        successful_frames = 0
        
        while self.video_streaming:
            try:
                # Get frame from Tello
                frame = self.controller.get_frame()
                frame_count += 1
                
                if frame is not None:
                    successful_frames += 1
                    
                    # Resize frame to fit display (Tello default is 960x720, we want 640x480)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(rgb_frame, (640, 480))
                    
                    
                    # Convert RGB to PIL Image (frame is already RGB from controller)
                    pil_image = Image.fromarray(frame_resized)
                    
                    # Convert to PhotoImage for tkinter
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update canvas on main thread
                    self.root.after(0, lambda p=photo: self.update_video_display(p))
                    
                    # Log success on first frame
                    if successful_frames == 1:
                        self.root.after(0, lambda: self.log_message(f"✅ Video frames flowing! ({frame_resized.shape[1]}x{frame_resized.shape[0]})"))
                        
                else:
                    # Handle empty frames
                    if frame_count % 30 == 0:  # Log every 30 attempts
                        self.root.after(0, lambda: self.log_message(f"⚠️ No video frame ({frame_count} attempts, {successful_frames} successful)"))
                
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.log_message(f"❌ Video loop error: {err}"))
                time.sleep(0.5)  # Wait longer on error
                
        # Log when video stops
        self.root.after(0, lambda: self.log_message(f"📹 Video loop ended. Total frames: {successful_frames}/{frame_count}"))
    
    def update_video_display(self, photo):
        """Update video display on main thread."""
        try:
            # Clear canvas and draw new frame
            self.video_canvas.delete("all")
            self.video_canvas.create_image(320, 240, image=photo)
            
            # Keep a reference to prevent garbage collection
            self.video_canvas.image = photo
            
        except Exception as e:
            self.log_message(f"Video display error: {e}")
    
    # Control methods
    def connect_drone(self):
        """Connect to the Tello drone."""
        try:
            self.log_message("Attempting to connect to Tello...")
            self.status_label.config(text="Connecting to Tello...")
            
            if self.controller.connect():
                self.log_message("Successfully connected to Tello!")
                self.status_label.config(text="Connected to Tello - Ready for flight")
            else:
                self.log_message("Failed to connect to Tello")
                self.status_label.config(text="Connection failed - Check Tello WiFi")
                
        except Exception as e:
            self.log_message(f"Connection error: {e}")
            self.status_label.config(text="Connection error")
    
    def disconnect_drone(self):
        """Disconnect from the Tello drone."""
        try:
            self.stop_monitoring()
            self.controller.disconnect()
            self.log_message("Disconnected from Tello")
            self.status_label.config(text="Disconnected")
            
        except Exception as e:
            self.log_message(f"Disconnect error: {e}")
    
    def start_monitoring(self):
        """Start the sensor monitoring."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.data_update_loop, daemon=True)
            self.update_thread.start()
            self.log_message("Started sensor monitoring")
            self.status_label.config(text="Monitoring active")
    
    def stop_monitoring(self):
        """Stop the sensor monitoring."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        self.log_message("Stopped sensor monitoring")
        self.status_label.config(text="Monitoring stopped")
    
    def takeoff_drone(self):
        """Command drone to takeoff."""
        try:
            self.log_message("Takeoff command sent")
            self.controller.takeoff()
        except Exception as e:
            self.log_message(f"Takeoff error: {e}")
    
    def land_drone(self):
        """Command drone to land."""
        try:
            self.log_message("Land command sent")
            self.controller.land()
        except Exception as e:
            self.log_message(f"Land error: {e}")
    
    def emergency_stop(self):
        """Emergency stop the drone."""
        try:
            self.log_message("EMERGENCY STOP ACTIVATED!")
            self.controller.emergency_stop()
        except Exception as e:
            self.log_message(f"Emergency stop error: {e}")
    
    def move_drone(self, direction):
        """Move drone in specified direction."""
        try:
            if direction == 'up':
                self.controller.move_up()
            elif direction == 'down':
                self.controller.move_down()
            elif direction == 'left':
                self.controller.move_left()
            elif direction == 'right':
                self.controller.move_right()
            elif direction == 'forward':
                self.controller.move_forward()
            elif direction == 'back':
                self.controller.move_back()
            
            self.log_message(f"Move {direction} command sent")
        except Exception as e:
            self.log_message(f"Move {direction} error: {e}")
    
    def rotate_drone(self, direction):
        """Rotate drone in specified direction."""
        try:
            if direction == 'cw':
                self.controller.rotate_clockwise()
                self.log_message("Rotate clockwise command sent")
            elif direction == 'ccw':
                self.controller.rotate_counter_clockwise()
                self.log_message("Rotate counter-clockwise command sent")
        except Exception as e:
            self.log_message(f"Rotate error: {e}")
    
    def toggle_video(self):
        """Toggle video streaming on/off."""
        try:
            if not self.video_streaming:
                # Check if drone is connected first
                if not self.controller.connected:
                    self.log_message("❌ Cannot start video - drone not connected")
                    messagebox.showerror("Error", "Please connect to Tello first before starting video")
                    return
                
                # Start video streaming
                self.log_message("📹 Starting video stream...")
                success = self.controller.start_video()
                
                if success:
                    self.video_streaming = True
                    self.video_thread = threading.Thread(target=self.video_update_loop, daemon=True)
                    self.video_thread.start()
                    
                    self.video_btn.config(text="📹 Stop Video", bg=self.colors['accent_red'])
                    self.video_status_text.config(text="Video: ON")
                    self.log_message("✅ Video streaming started successfully")
                else:
                    self.log_message("❌ Failed to start video streaming")
                    self.log_message("💡 Try: 1) Restart drone 2) Check WiFi 3) Close other apps")
                    messagebox.showerror("Video Error", 
                                       "Failed to start video streaming.\n\n" +
                                       "Common fixes:\n" +
                                       "• Restart the Tello drone\n" +
                                       "• Check WiFi connection\n" +
                                       "• Close other apps using the camera\n" +
                                       "• Try disconnecting and reconnecting")
            else:
                # Stop video streaming
                self.log_message("📹 Stopping video stream...")
                self.video_streaming = False
                if self.video_thread:
                    self.video_thread.join(timeout=2)
                
                self.controller.stop_video()
                
                # Clear video display
                self.video_canvas.delete("all")
                self.video_canvas.create_text(320, 240, text="📹 Video Stream Inactive\n\nConnect to Tello and click\n'Start Video' to begin streaming", 
                                            fill='white', font=('Arial', 14), justify='center')
                
                self.video_btn.config(text="📹 Start Video", bg=self.colors['bg_secondary'])
                self.video_status_text.config(text="Video: OFF")
                self.log_message("✅ Video streaming stopped")
                
        except Exception as e:
            self.log_message(f"❌ Video toggle error: {e}")
            messagebox.showerror("Error", f"Video error: {e}")
    
    def save_data(self):
        """Save current sensor data to file."""
        if not self.sensor_data:
            messagebox.showwarning("No Data", "No sensor data available to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Sensor Data"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.sensor_data, f, indent=2)
                self.log_message(f"Data saved to {filename}")
                messagebox.showinfo("Success", f"Data saved to {filename}")
            except Exception as e:
                self.log_message(f"Save error: {e}")
                messagebox.showerror("Error", f"Failed to save data: {e}")
    
    def run(self):
        """Start the HUD application."""
        self.log_message("Tello HUD initialized")
        self.status_label.config(text="Ready - Connect to Tello to begin")
        
        # Handle window close
        def on_closing():
            self.stop_monitoring()
            if self.video_streaming:
                self.video_streaming = False
                if self.video_thread:
                    self.video_thread.join(timeout=1)
                self.controller.stop_video()
            self.controller.disconnect()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

def main():
    """Main entry point."""
    print("🚁 Starting Tello HUD...")
    print("Make sure your Tello is powered on and connected to WiFi")
    
    try:
        hud = TelloHUD()
        hud.run()
    except Exception as e:
        print(f"❌ Failed to start HUD: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 