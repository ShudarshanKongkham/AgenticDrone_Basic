import cv2
import numpy as np
import pygame
import plotly.graph_objects as go
import plotly.io as pio
import threading
import time
from queue import Queue
import warnings
import io
import base64
from PIL import Image
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

warnings.filterwarnings('ignore')

# Import the necessary components
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

class ManeuverAction(Enum):
    """Enum for different maneuvering actions"""
    MOVE_FORWARD = "move_forward"
    ROTATE_CLOCKWISE = "rotate_clockwise"
    ROTATE_COUNTERCLOCKWISE = "rotate_counterclockwise"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    HOVER = "hover"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class NavigationAssessment:
    """Data class for navigation assessment results"""
    is_safe_to_proceed: bool
    recommended_action: ManeuverAction
    confidence: float
    obstacle_distance: float
    safe_path_angle: Optional[float]  # Angle in degrees for safe path
    rotation_direction: Optional[str]  # "clockwise" or "counterclockwise"
    maneuver_steps: List[str]
    risk_level: str
    alternative_paths: List[Dict]

@dataclass
class FlightPath:
    """Data class for flight path visualization"""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    waypoints: List[Tuple[int, int]]
    safety_score: float
    path_type: str  # "direct", "curved", "multi_step"

class RealTimeDroneAnalyzer:
    """Real-time drone navigation analyzer using webcam input with single display"""
    
    def __init__(self, window_width=1200, window_height=600):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Safety thresholds - More realistic for drone navigation
        self.critical_distance = 0.5   # Critical danger - immediate stop
        self.danger_distance = 1.0     # Close obstacle - avoid direction
        self.warning_distance = 1.8    # Warning zone - proceed with caution
        self.safe_distance = 3.0       # Safe zone - clear path
        
        # Navigation settings
        self.min_safe_percentage = 60  # Minimum % of safe pixels to consider direction viable
        self.emergency_threshold = 0.3 # If ANY direction has obstacle closer than this, emergency stop
        
        # Display settings
        self.window_width = window_width
        self.window_height = window_height
        self.video_width = window_width // 2
        self.video_height = window_height
        self.plot_width = window_width // 2
        self.plot_height = window_height
        
        # Webcam settings
        self.cap = None
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Pygame setup
        pygame.init()
        self.screen = None
        
        # Current analysis data
        self.current_analysis = None
        self.analysis_lock = threading.Lock()
        
        # Navigation planner
        self.navigation_planner = DroneNavigationPlanner(self)
        self.current_navigation_assessment = None
        
        # Load depth estimation model
        self.load_depth_model()
        
        # Initialize fonts for text rendering
        pygame.init()  # Ensure pygame is initialized
        try:
            self.small_font = pygame.font.Font(None, 18)
            self.font = pygame.font.Font(None, 24)
        except Exception as e:
            print(f"Warning: Could not initialize fonts: {e}")
            # Use default font as fallback
            default_font = pygame.font.get_default_font()
            self.small_font = pygame.font.Font(default_font, 18)
            self.font = pygame.font.Font(default_font, 24)
        
    def load_depth_model(self):
        """Load the depth estimation model"""
        try:
            print("📥 Loading depth estimation model...")
            model_id = "Intel/dpt-large"
            
            self.processor = DPTImageProcessor.from_pretrained(model_id)
            self.model = DPTForDepthEstimation.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Depth model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load depth model: {str(e)}")
            # Fallback to a simpler model
            try:
                print("🔄 Trying fallback model...")
                model_id = "Intel/dpt-hybrid-midas"
                self.processor = DPTImageProcessor.from_pretrained(model_id)
                self.model = DPTForDepthEstimation.from_pretrained(model_id)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Fallback model loaded successfully!")
            except Exception as e2:
                print(f"❌ Failed to load fallback model: {str(e2)}")
                raise e2
    
    def predict_depth(self, image):
        """Predict depth map from image"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            # Predict depth
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Convert to numpy and normalize
            depth_map = predicted_depth.squeeze().cpu().numpy()
            
            return depth_map
            
        except Exception as e:
            print(f"Error in depth prediction: {e}")
            return None
    
    def calibrate_depth_to_distance(self, depth_map, min_distance=0.3, max_distance=8.0):
        """Calibrate relative depth values to real-world distances for indoor navigation"""
        if depth_map.min() < 0:
            depth_map = -depth_map
        
        # Normalize to 0-1 range
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Map to distance range (more suitable for indoor drone navigation)
        distance_map = min_distance + (1 - depth_normalized) * (max_distance - min_distance)
        
        return distance_map
    
    def analyze_flight_path(self, distance_map):
        """Analyze the image for safe flight directions in all 6 directions"""
        h, w = distance_map.shape
        center_h, center_w = h // 2, w // 2
        
        # Define 6 primary sectors for drone movement
        sectors = {
            'forward': distance_map[center_h-h//6:center_h+h//6, 2*w//3:],  # Forward (right side of image)
            'backward': distance_map[center_h-h//6:center_h+h//6, :w//3],   # Backward (left side of image)
            'left': distance_map[center_h-h//4:center_h+h//4, w//3:2*w//3], # Left (center-left)
            'right': distance_map[center_h-h//4:center_h+h//4, w//3:2*w//3], # Right (center-right)
            'up': distance_map[:h//3, center_w-w//4:center_w+w//4],         # Up (top section)
            'down': distance_map[2*h//3:, center_w-w//4:center_w+w//4]      # Down (bottom section)
        }
        
        # Analyze each sector
        analysis = {}
        for direction, sector in sectors.items():
            min_dist = float(sector.min())
            mean_dist = float(sector.mean())
            
            # Calculate safety percentage (pixels beyond safe distance)
            safe_pixels = np.sum(sector >= self.safe_distance)
            safe_percentage = (safe_pixels / sector.size) * 100
            
            # Determine safety level based on minimum distance
            if min_dist < self.critical_distance:
                safety_level = "CRITICAL"
                recommendation = "EMERGENCY_AVOID"
            elif min_dist < self.danger_distance:
                safety_level = "DANGER"
                recommendation = "AVOID"
            elif min_dist < self.warning_distance:
                safety_level = "WARNING"
                recommendation = "CAUTION"
            elif min_dist < self.safe_distance:
                safety_level = "CAUTION"
                recommendation = "PROCEED_CAREFULLY"
            else:
                safety_level = "SAFE"
                recommendation = "CLEAR"
            
            # Factor in safe percentage for overall viability
            if safe_percentage < self.min_safe_percentage and safety_level != "CRITICAL":
                if safety_level == "SAFE":
                    safety_level = "CAUTION"
                    recommendation = "PROCEED_CAREFULLY"
                elif safety_level == "CAUTION":
                    safety_level = "WARNING"
                    recommendation = "CAUTION"
            
            analysis[direction] = {
                'min_distance': min_dist,
                'mean_distance': mean_dist,
                'safety_level': safety_level,
                'recommendation': recommendation,
                'safe_percentage': safe_percentage,
                'is_viable': safe_percentage >= self.min_safe_percentage and min_dist >= self.danger_distance
            }
        
        # Add overall recommendation
        analysis['overall'] = self._get_overall_recommendation(analysis)
        
        return analysis
    
    def _get_overall_recommendation(self, analysis):
        """Generate overall flight recommendation based on all directions"""
        directions = ['forward', 'backward', 'left', 'right', 'up', 'down']
        
        # Check for emergency situations first
        closest_obstacle = min(analysis[d]['min_distance'] for d in directions)
        
        # Emergency stop only if critically close obstacle
        if closest_obstacle < self.emergency_threshold:
            return {
                'recommendation': "EMERGENCY_STOP",
                'safest_direction': 'none',
                'closest_obstacle': closest_obstacle,
                'confidence': 0.95,
                'reason': f"Critical obstacle at {closest_obstacle:.2f}m"
            }
        
        # Find viable directions (safe enough to navigate)
        viable_directions = [d for d in directions if analysis[d]['is_viable']]
        
        if not viable_directions:
            # No viable directions - find least dangerous
            safest_direction = max(directions, key=lambda d: analysis[d]['min_distance'])
            
            if analysis[safest_direction]['min_distance'] < self.danger_distance:
                recommendation = "HOVER_AND_REASSESS"
                confidence = 0.4
            else:
                recommendation = f"SLOW_TURN_{safest_direction.upper()}"
                confidence = 0.6
        else:
            # Find best viable direction
            # Prioritize forward movement, then by safety score
            def safety_score(direction):
                data = analysis[direction]
                score = data['min_distance'] * 0.6 + (data['safe_percentage'] / 100) * 0.4
                if direction == 'forward':  # Prefer forward movement
                    score += 0.5
                elif direction in ['up', 'down']:  # Deprioritize vertical movement
                    score -= 0.2
                return score
            
            best_direction = max(viable_directions, key=safety_score)
            best_data = analysis[best_direction]
            
            if best_data['safety_level'] == 'SAFE':
                if best_direction == 'forward':
                    recommendation = "CONTINUE_FORWARD"
                else:
                    recommendation = f"NAVIGATE_{best_direction.upper()}"
                confidence = 0.8
            elif best_data['safety_level'] == 'CAUTION':
                recommendation = f"CAREFUL_{best_direction.upper()}"
                confidence = 0.6
            else:
                recommendation = f"SLOW_{best_direction.upper()}"
                confidence = 0.5
        
        return {
            'recommendation': recommendation,
            'safest_direction': viable_directions[0] if viable_directions else max(directions, key=lambda d: analysis[d]['min_distance']),
            'closest_obstacle': closest_obstacle,
            'confidence': confidence,
            'viable_directions': len(viable_directions),
            'reason': f"{len(viable_directions)} viable paths"
        }

    def create_plotly_radar_chart(self, analysis):
        """Create a Plotly radar chart similar to the reference image"""
        if not analysis:
            return None
        
        # Primary directions for radar chart
        directions = ['forward', 'right', 'backward', 'left', 'up', 'down']
        
        # Extract data for each direction
        min_distances = []
        safety_percentages = []
        direction_labels = []
        
        for direction in directions:
            if direction in analysis:
                min_distances.append(analysis[direction]['min_distance'])
                safety_percentages.append(analysis[direction]['safe_percentage'] / 20)  # Scale for visualization
                direction_labels.append(direction.title())
            else:
                min_distances.append(0)
                safety_percentages.append(0)
                direction_labels.append(direction.title())
        
        # Create radar chart
        fig = go.Figure()
        
        # Add Min Distance trace (blue)
        fig.add_trace(go.Scatterpolar(
            r=min_distances,
            theta=direction_labels,
            fill='toself',
            name='Min Distance (m)',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.3)',
            opacity=0.8
        ))
        
        # Add Safety % trace (green, scaled)
        fig.add_trace(go.Scatterpolar(
            r=safety_percentages,
            theta=direction_labels,
            fill='toself',
            name='Safety % (scaled)',
            line_color='#2ca02c',
            fillcolor='rgba(44, 160, 44, 0.3)',
            opacity=0.8
        ))
        
        # Update layout to match reference style
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(min_distances), max(safety_percentages)) * 1.1],
                    showticklabels=True,
                    tickfont=dict(size=10),
                    gridcolor='rgba(255, 255, 255, 0.3)'
                ),
                angularaxis=dict(
                    showticklabels=True,
                    tickfont=dict(size=12),
                    gridcolor='rgba(255, 255, 255, 0.3)'
                ),
                bgcolor='rgba(0, 0, 0, 0)'
            ),
            showlegend=True,
            title=dict(
                text="🧭 Directional Safety Analysis",
                x=0.5,
                font=dict(size=16, color='white')
            ),
            font=dict(color='white'),
            paper_bgcolor='rgba(0, 0, 0, 0.8)',
            plot_bgcolor='rgba(0, 0, 0, 0.8)',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            width=400,
            height=400
        )
        
        return fig

    def draw_radar_plot(self, surface, analysis, x_offset=0, y_offset=0):
        """Draw radar plot directly on pygame surface with improved styling"""
        center_x = x_offset + self.plot_width // 2
        center_y = y_offset + self.plot_height // 2
        radius = min(self.plot_width, self.plot_height) // 3 - 20
        
        # Clear the plot area with dark background
        plot_rect = pygame.Rect(x_offset, y_offset, self.plot_width, self.plot_height)
        pygame.draw.rect(surface, (15, 15, 25), plot_rect)
        
        # Draw concentric circles for distance reference
        circle_colors = [(40, 40, 60), (60, 60, 80), (80, 80, 100), (100, 100, 120)]
        for i, color in enumerate(circle_colors):
            circle_radius = radius * (i + 1) // 4
            pygame.draw.circle(surface, color, (center_x, center_y), circle_radius, 1)
        
        # Six primary directions with angles (in radians)
        directions = {
            'forward': 0,              # Right (0°)
            'right': math.pi/3,        # Bottom-right (60°)
            'backward': math.pi,       # Left (180°)
            'left': 4*math.pi/3,       # Top-left (240°)
            'up': 5*math.pi/3,         # Top-right (300°)
            'down': 2*math.pi/3        # Bottom-left (120°)
        }
        
        if analysis:
            # Prepare data for both traces
            min_distances = []
            safety_percentages = []
            angles = []
            
            for direction, angle in directions.items():
                if direction in analysis:
                    min_dist = analysis[direction]['min_distance']
                    safety_pct = analysis[direction]['safe_percentage']
                    
                    min_distances.append((angle, min_dist))
                    safety_percentages.append((angle, safety_pct / 20))  # Scale for visualization
                    angles.append(angle)
            
            # Sort by angle for proper polygon drawing
            min_distances.sort(key=lambda x: x[0])
            safety_percentages.sort(key=lambda x: x[0])
            
            # Draw Min Distance trace (blue)
            min_dist_points = []
            for angle, dist in min_distances:
                dist_normalized = min(dist / 10.0, 1.0)  # Max 10m
                point_radius = int(dist_normalized * radius)
                point_x = center_x + int(point_radius * math.cos(angle))
                point_y = center_y + int(point_radius * math.sin(angle))
                min_dist_points.append((point_x, point_y))
            
            # Draw filled polygon for min distance
            if len(min_dist_points) >= 3:
                pygame.draw.polygon(surface, (31, 119, 180, 80), min_dist_points)
                pygame.draw.polygon(surface, (31, 119, 180), min_dist_points, 2)
            
            # Draw Safety % trace (green)
            safety_points = []
            for angle, safety in safety_percentages:
                safety_normalized = min(safety, 1.0)
                point_radius = int(safety_normalized * radius)
                point_x = center_x + int(point_radius * math.cos(angle))
                point_y = center_y + int(point_radius * math.sin(angle))
                safety_points.append((point_x, point_y))
            
            # Draw filled polygon for safety percentage
            if len(safety_points) >= 3:
                # Create a transparent surface for the safety overlay
                safety_surface = pygame.Surface((self.plot_width, self.plot_height), pygame.SRCALPHA)
                pygame.draw.polygon(safety_surface, (44, 160, 44, 60), 
                                  [(x - x_offset, y - y_offset) for x, y in safety_points])
                surface.blit(safety_surface, (x_offset, y_offset))
                pygame.draw.polygon(surface, (44, 160, 44), safety_points, 2)
            
            # Draw direction labels and values
            for direction, angle in directions.items():
                if direction in analysis:
                    min_dist = analysis[direction]['min_distance']
                    safety_level = analysis[direction]['safety_level']
                    
                    # Calculate label position
                    label_radius = radius + 40
                    label_x = center_x + int(label_radius * math.cos(angle))
                    label_y = center_y + int(label_radius * math.sin(angle))
                    
                    # Color based on safety level
                    if safety_level == "CRITICAL":
                        color = (255, 20, 20)  # Bright red
                    elif safety_level == "DANGER":
                        color = (255, 60, 60)
                    elif safety_level == "WARNING":
                        color = (255, 165, 0)
                    elif safety_level == "CAUTION":
                        color = (255, 255, 100)
                    else:
                        color = (100, 255, 100)
                    
                    # Add viable indicator
                    is_viable = analysis[direction].get('is_viable', False)
                    if is_viable:
                        # Draw green outline for viable directions
                        pygame.draw.circle(surface, (0, 255, 0), (label_x, label_y), 25, 2)
                    
                    # Draw direction line
                    line_end_x = center_x + int((radius - 10) * math.cos(angle))
                    line_end_y = center_y + int((radius - 10) * math.sin(angle))
                    pygame.draw.line(surface, (120, 120, 120), (center_x, center_y), 
                                   (line_end_x, line_end_y), 1)
                    
                    # Draw direction label
                    label_text = f"{direction.upper()}"
                    text = self.small_font.render(label_text, True, color)
                    text_rect = text.get_rect()
                    text_rect.center = (label_x, label_y - 8)
                    surface.blit(text, text_rect)
                    
                    # Draw distance value
                    dist_text = f"{min_dist:.1f}m"
                    dist_surface = self.small_font.render(dist_text, True, (200, 200, 200))
                    dist_rect = dist_surface.get_rect()
                    dist_rect.center = (label_x, label_y + 8)
                    surface.blit(dist_surface, dist_rect)
        
        # Draw center point
        pygame.draw.circle(surface, (255, 255, 255), (center_x, center_y), 3)
        
        # Draw title
        title_text = self.font.render(f"🧭 Directional Safety Analysis (FPS: {self.current_fps:.1f})", True, (255, 255, 255))
        title_rect = title_text.get_rect()
        title_rect.centerx = center_x
        title_rect.y = y_offset + 10
        surface.blit(title_text, title_rect)
        
        # Draw legend
        legend_y = y_offset + self.plot_height - 120
        legend_x = x_offset + 20
        
        # Legend background
        legend_bg = pygame.Rect(legend_x - 10, legend_y - 10, 200, 100)
        pygame.draw.rect(surface, (20, 20, 30, 180), legend_bg)
        pygame.draw.rect(surface, (100, 100, 120), legend_bg, 1)
        
        # Legend items
        legend_items = [
            ("Min Distance (m)", (31, 119, 180)),
            ("Safety % (scaled)", (44, 160, 44)),
            ("", (0, 0, 0)),  # Spacer
            ("SAFE", (100, 255, 100)),
            ("CAUTION", (255, 255, 100)),
            ("WARNING", (255, 165, 0)),
            ("DANGER", (255, 60, 60)),
            ("CRITICAL", (255, 20, 20)),
            ("", (0, 0, 0)),  # Spacer
            ("Viable Path", (0, 255, 0))
        ]
        
        for i, (label, color) in enumerate(legend_items):
            if label:  # Skip empty spacer
                y_pos = legend_y + i * 12
                if "Distance" in label or "Safety" in label:
                    # Draw line for traces
                    pygame.draw.line(surface, color, (legend_x, y_pos + 4), (legend_x + 15, y_pos + 4), 3)
                elif "Viable" in label:
                    # Draw circle outline for viable paths
                    pygame.draw.circle(surface, color, (legend_x + 8, y_pos + 4), 6, 2)
                else:
                    # Draw filled circle for safety levels
                    pygame.draw.circle(surface, color, (legend_x + 8, y_pos + 4), 6)
                
                text = self.small_font.render(label, True, (255, 255, 255))
                surface.blit(text, (legend_x + 25, y_pos - 2))
        
        # Draw overall recommendation if available
        if analysis and 'overall' in analysis:
            overall = analysis['overall']
            rec_y = y_offset + 50
            
            rec_bg = pygame.Rect(x_offset + 10, rec_y, self.plot_width - 20, 80)
            pygame.draw.rect(surface, (30, 30, 40, 180), rec_bg)
            pygame.draw.rect(surface, (100, 100, 120), rec_bg, 1)
            
            # Recommendation text
            rec_text = overall['recommendation'].replace('_', ' ')
            confidence = overall['confidence']
            viable_count = overall.get('viable_directions', 0)
            reason = overall.get('reason', '')
            
            # Color based on recommendation
            if "EMERGENCY" in rec_text:
                rec_color = (255, 60, 60)
            elif "HOVER" in rec_text or "SLOW" in rec_text:
                rec_color = (255, 165, 0)
            elif "CAREFUL" in rec_text:
                rec_color = (255, 255, 100)
            else:
                rec_color = (100, 255, 100)
            
            rec_surface = self.font.render(f"🚁 {rec_text}", True, rec_color)
            surface.blit(rec_surface, (x_offset + 20, rec_y + 8))
            
            conf_surface = self.small_font.render(f"Confidence: {confidence:.1%} | Viable: {viable_count}/6", True, (200, 200, 200))
            surface.blit(conf_surface, (x_offset + 20, rec_y + 32))
            
            closest_surface = self.small_font.render(f"Closest: {overall['closest_obstacle']:.1f}m | {reason}", True, (200, 200, 200))
            surface.blit(closest_surface, (x_offset + 20, rec_y + 48))
            
            # Add threshold reference
            threshold_text = f"Thresholds: Critical<{self.emergency_threshold}m, Danger<{self.danger_distance}m, Safe>{self.safe_distance}m"
            threshold_surface = self.small_font.render(threshold_text, True, (150, 150, 150))
            surface.blit(threshold_surface, (x_offset + 20, rec_y + 64))
    
    def process_frame(self, frame):
        """Process a single frame and return analysis results"""
        try:
            # Resize frame for faster processing
            processing_size = (320, 240)  # Smaller for better performance
            resized_frame = cv2.resize(frame, processing_size)
            
            # Predict depth
            depth_map = self.predict_depth(resized_frame)
            if depth_map is None:
                return None
            
            # Calibrate distances
            distance_map = self.calibrate_depth_to_distance(depth_map, 0.3, 8.0)
            
            # Analyze flight path
            flight_analysis = self.analyze_flight_path(distance_map)
            
            # Get navigation assessment for forward movement
            navigation_assessment = self.navigation_planner.assess_forward_movement(distance_map)
            
            return {
                'flight_analysis': flight_analysis,
                'distance_map': distance_map,
                'navigation_assessment': navigation_assessment
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def frame_capture_thread(self):
        """Capture frames from webcam in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(0.03)  # ~30 FPS
    
    def frame_processing_thread(self):
        """Process frames in a separate thread"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    result = self.process_frame(frame)
                    
                    if result is not None:
                        with self.analysis_lock:
                            self.current_analysis = result
                        
                        # Update FPS counter
                        self.fps_counter += 1
                        current_time = time.time()
                        if current_time - self.fps_start_time >= 1.0:
                            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                            self.fps_counter = 0
                            self.fps_start_time = current_time
                
                time.sleep(0.1)  # Process every 100ms
                
            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(0.5)
    
    def start_analysis(self, camera_index=0):
        """Start real-time analysis"""
        try:
            # Initialize webcam
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"📷 Camera {camera_index} initialized successfully!")
            print("🚀 Starting real-time analysis...")
            print("Press ESC to quit, SPACE to save current analysis")
            
            # Initialize pygame window
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("🚁 Real-Time Drone Navigation System")
            
            self.running = True
            
            # Start capture and processing threads
            capture_thread = threading.Thread(target=self.frame_capture_thread)
            processing_thread = threading.Thread(target=self.frame_processing_thread)
            
            capture_thread.daemon = True
            processing_thread.daemon = True
            
            capture_thread.start()
            processing_thread.start()
            
            clock = pygame.time.Clock()
            
            # Main display loop
            while self.running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            # Save current analysis
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            print(f"💾 Analysis snapshot saved at {timestamp}")
                            
                            # Save Plotly chart if analysis is available
                            with self.analysis_lock:
                                current_analysis_copy = self.current_analysis
                            
                            if current_analysis_copy:
                                try:
                                    # Create and save Plotly radar chart
                                    fig = self.create_plotly_radar_chart(current_analysis_copy['flight_analysis'])
                                    if fig:
                                        chart_filename = f"radar_chart_{timestamp}.html"
                                        fig.write_html(chart_filename)
                                        print(f"📊 Plotly radar chart saved as {chart_filename}")
                                        
                                        # Also save as PNG if kaleido is available
                                        try:
                                            png_filename = f"radar_chart_{timestamp}.png"
                                            fig.write_image(png_filename, width=800, height=600)
                                            print(f"🖼️ PNG chart saved as {png_filename}")
                                        except Exception as e:
                                            print(f"⚠️ PNG save failed (install kaleido for PNG export): {e}")
                                    
                                    # Save analysis data as JSON
                                    import json
                                    analysis_filename = f"analysis_data_{timestamp}.json"
                                    with open(analysis_filename, 'w') as f:
                                        # Convert numpy arrays to lists for JSON serialization
                                        json_data = {}
                                        for direction, data in current_analysis_copy['flight_analysis'].items():
                                            if isinstance(data, dict):
                                                json_data[direction] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                                                      for k, v in data.items()}
                                        json.dump(json_data, f, indent=2)
                                    print(f"📄 Analysis data saved as {analysis_filename}")
                                    
                                except Exception as e:
                                    print(f"❌ Error saving files: {e}")
                
                # Clear screen
                self.screen.fill((0, 0, 0))
                
                # Get and display webcam frame
                ret, frame = self.cap.read()
                if ret:
                    # Resize frame to fit left half of window
                    frame_resized = cv2.resize(frame, (self.video_width, self.video_height))
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    # Convert to pygame surface
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    
                    # Blit to screen
                    self.screen.blit(frame_surface, (0, 0))
                    
                    # Get current analysis data
                    with self.analysis_lock:
                        current_analysis_copy = self.current_analysis
                    
                    # Draw navigation overlay if available
                    if current_analysis_copy and 'navigation_assessment' in current_analysis_copy:
                        nav_assessment = current_analysis_copy['navigation_assessment']
                        distance_map = current_analysis_copy['distance_map']
                        
                        # Draw navigation paths and guidance on the VIDEO FEED (left side)
                        self.navigation_planner.draw_navigation_overlay(
                            frame_surface, distance_map, nav_assessment, 0, 0
                        )
                        
                        # Re-blit the frame with navigation overlay
                        self.screen.blit(frame_surface, (0, 0))
                        
                        # Get detailed guidance text
                        guidance_text = self.navigation_planner.get_detailed_guidance(nav_assessment)
                        
                        # Display navigation guidance overlay on video feed
                        guidance_bg = pygame.Surface((self.video_width - 20, 120), pygame.SRCALPHA)
                        guidance_bg.fill((0, 0, 0, 180))
                        self.screen.blit(guidance_bg, (10, self.video_height - 140))
                        
                        # Draw guidance text
                        guidance_lines = guidance_text.split(" | ")
                        for i, line in enumerate(guidance_lines[:5]):  # Show max 5 lines
                            if line:
                                text_surface = self.small_font.render(line, True, (255, 255, 255))
                                self.screen.blit(text_surface, (20, self.video_height - 130 + i * 18))
                        
                        # Draw maneuver steps if available - positioned better
                        if nav_assessment and nav_assessment.maneuver_steps:
                            steps_bg = pygame.Surface((self.video_width - 20, 100), pygame.SRCALPHA)
                            steps_bg.fill((0, 20, 40, 160))
                            self.screen.blit(steps_bg, (10, 80))
                            
                            steps_title = self.font.render("🛠️ MANEUVER STEPS:", True, (0, 255, 255))
                            self.screen.blit(steps_title, (20, 90))
                            
                            for i, step in enumerate(nav_assessment.maneuver_steps[:4]):  # Show max 4 steps
                                step_text = self.small_font.render(f"{i+1}. {step}", True, (200, 255, 200))
                                self.screen.blit(step_text, (20, 115 + i * 16))
                    
                    # Add overlay text on video with better styling
                    overlay_bg = pygame.Surface((250, 80), pygame.SRCALPHA)
                    overlay_bg.fill((0, 0, 0, 128))
                    self.screen.blit(overlay_bg, (10, 10))
                    
                    fps_text = self.font.render(f"FPS: {self.current_fps:.1f}", True, (0, 255, 0))
                    self.screen.blit(fps_text, (20, 20))
                    
                    # Display current analysis summary
                    if current_analysis_copy and 'flight_analysis' in current_analysis_copy:
                        overall = current_analysis_copy['flight_analysis'].get('overall', {})
                        if overall:
                            closest = overall.get('closest_obstacle', 0)
                            recommendation = overall.get('recommendation', 'ANALYZING').replace('_', ' ')
                            
                            closest_text = self.small_font.render(f"Closest: {closest:.1f}m", True, (255, 255, 255))
                            self.screen.blit(closest_text, (20, 45))
                            
                            rec_color = (255, 60, 60) if "EMERGENCY" in recommendation else (255, 165, 0) if "TURN" in recommendation else (100, 255, 100)
                            rec_text = self.small_font.render(f"Action: {recommendation[:20]}", True, rec_color)
                            self.screen.blit(rec_text, (20, 65))
                    
                    instructions = self.small_font.render("ESC: Quit | SPACE: Save Charts & Data", True, (255, 255, 255))
                    self.screen.blit(instructions, (10, self.video_height - 25))
                
                # Draw radar plot on right half
                with self.analysis_lock:
                    current_analysis = self.current_analysis
                
                if current_analysis:
                    self.draw_radar_plot(self.screen, current_analysis['flight_analysis'], 
                                       self.video_width, 0)
                else:
                    # Draw placeholder
                    placeholder_text = self.font.render("Processing...", True, (255, 255, 255))
                    placeholder_rect = placeholder_text.get_rect()
                    placeholder_rect.center = (self.video_width + self.plot_width//2, self.plot_height//2)
                    self.screen.blit(placeholder_text, placeholder_rect)
                
                # Update display
                pygame.display.flip()
                clock.tick(30)  # 30 FPS display
            
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
        
        finally:
            self.stop_analysis()
    
    def stop_analysis(self):
        """Stop the real-time analysis"""
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        pygame.quit()
        print("🛑 Analysis stopped")

class DroneNavigationPlanner:
    """Advanced navigation planner for obstacle avoidance and path planning"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.flight_paths = []
        self.current_assessment = None
        
        # Navigation parameters
        self.min_forward_distance = 2.0  # Minimum safe distance to move forward
        self.rotation_increment = 15     # Degrees to rotate per step
        self.max_rotation_search = 180   # Maximum rotation to search for safe path
        self.path_width = 1.5           # Required clear width for safe passage
        
        # Initialize fonts for text rendering
        pygame.init()  # Ensure pygame is initialized
        try:
            self.small_font = pygame.font.Font(None, 18)
            self.font = pygame.font.Font(None, 24)
        except Exception as e:
            print(f"Warning: Could not initialize fonts: {e}")
            # Use default font as fallback
            default_font = pygame.font.get_default_font()
            self.small_font = pygame.font.Font(default_font, 18)
            self.font = pygame.font.Font(default_font, 24)
        
    def assess_forward_movement(self, distance_map, current_heading=0) -> NavigationAssessment:
        """
        Assess if it's safe to move forward and provide maneuvering guidance
        
        Args:
            distance_map: Depth/distance map from the depth estimation
            current_heading: Current drone heading in degrees (0 = forward)
        
        Returns:
            NavigationAssessment with detailed guidance
        """
        h, w = distance_map.shape
        center_h, center_w = h // 2, w // 2
        
        # Define forward sector (more focused)
        forward_sector = distance_map[center_h-h//8:center_h+h//8, int(w*0.4):int(w*0.9)]
        
        # Check immediate forward path
        min_forward_dist = float(forward_sector.min())
        mean_forward_dist = float(forward_sector.mean())
        
        # Calculate safe percentage in forward direction
        safe_pixels = np.sum(forward_sector >= self.min_forward_distance)
        safe_percentage = (safe_pixels / forward_sector.size) * 100
        
        # Determine if forward movement is safe
        is_safe_forward = (min_forward_dist >= self.min_forward_distance and 
                          safe_percentage >= 70)
        
        if is_safe_forward:
            return NavigationAssessment(
                is_safe_to_proceed=True,
                recommended_action=ManeuverAction.MOVE_FORWARD,
                confidence=0.9,
                obstacle_distance=min_forward_dist,
                safe_path_angle=0.0,
                rotation_direction=None,
                maneuver_steps=["Move forward safely"],
                risk_level="LOW",
                alternative_paths=[]
            )
        
        # Forward path blocked - find alternative maneuvering strategy
        return self._find_maneuvering_strategy(distance_map, current_heading, min_forward_dist)
    
    def _find_maneuvering_strategy(self, distance_map, current_heading, forward_obstacle_dist) -> NavigationAssessment:
        """Find the best maneuvering strategy to avoid obstacles"""
        h, w = distance_map.shape
        center_h, center_w = h // 2, w // 2
        
        # Search for safe paths by checking different rotation angles
        safe_paths = []
        
        for angle_offset in range(-self.max_rotation_search//2, self.max_rotation_search//2, self.rotation_increment):
            if angle_offset == 0:  # Skip forward (already checked)
                continue
                
            # Calculate sector for this angle
            path_info = self._assess_path_at_angle(distance_map, angle_offset)
            if path_info['is_safe']:
                safe_paths.append(path_info)
        
        if not safe_paths:
            # No safe paths found - emergency situation
            return NavigationAssessment(
                is_safe_to_proceed=False,
                recommended_action=ManeuverAction.EMERGENCY_STOP,
                confidence=0.9,
                obstacle_distance=forward_obstacle_dist,
                safe_path_angle=None,
                rotation_direction=None,
                maneuver_steps=["EMERGENCY: No safe paths available", "Hover and reassess"],
                risk_level="CRITICAL",
                alternative_paths=[]
            )
        
        # Find the best safe path
        best_path = max(safe_paths, key=lambda x: x['safety_score'])
        
        # Determine rotation direction and steps
        rotation_direction = "clockwise" if best_path['angle'] > 0 else "counterclockwise"
        rotation_steps = abs(best_path['angle']) // self.rotation_increment
        
        # Generate maneuvering steps
        maneuver_steps = self._generate_maneuver_steps(best_path, rotation_steps)
        
        # Determine primary action
        if abs(best_path['angle']) <= 30:
            recommended_action = ManeuverAction.ROTATE_CLOCKWISE if best_path['angle'] > 0 else ManeuverAction.ROTATE_COUNTERCLOCKWISE
        else:
            recommended_action = ManeuverAction.HOVER  # Need significant reorientation
        
        return NavigationAssessment(
            is_safe_to_proceed=True,
            recommended_action=recommended_action,
            confidence=best_path['safety_score'],
            obstacle_distance=forward_obstacle_dist,
            safe_path_angle=float(best_path['angle']),
            rotation_direction=rotation_direction,
            maneuver_steps=maneuver_steps,
            risk_level="MEDIUM" if abs(best_path['angle']) <= 45 else "HIGH",
            alternative_paths=safe_paths[:3]  # Top 3 alternatives
        )
    
    def _assess_path_at_angle(self, distance_map, angle_degrees):
        """Assess safety of a path at given angle offset"""
        h, w = distance_map.shape
        center_h, center_w = h // 2, w // 2
        
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        
        # Create a cone-shaped sector extending in the direction of the angle
        sector_width = w // 8  # Narrower width for more precise path checking
        sector_length = min(h, w) // 4  # Reasonable forward projection
        
        # Generate points along the path
        path_points = []
        path_distances = []
        
        # Sample multiple points along the intended path
        for dist in range(15, sector_length, 8):
            # Calculate center point of path at this distance
            dx = int(dist * math.sin(angle_rad))
            dy = int(dist * math.cos(angle_rad))
            
            center_x = center_w + dx
            center_y = center_h - dy  # Negative because image y increases downward
            
            # Check if point is within image bounds
            if 0 <= center_x < w and 0 <= center_y < h:
                path_points.append((center_x, center_y))
                
                # Sample in a cross pattern around each point for better obstacle detection
                sample_distances = []
                for sample_dx in [-sector_width//6, 0, sector_width//6]:
                    for sample_dy in [-3, 0, 3]:
                        sample_x = center_x + sample_dx
                        sample_y = center_y + sample_dy
                        
                        if 0 <= sample_x < w and 0 <= sample_y < h:
                            sample_distances.append(distance_map[sample_y, sample_x])
                
                if sample_distances:
                    # Use minimum distance in the sampled area
                    path_distances.append(min(sample_distances))
        
        if not path_distances:
            return {'is_safe': False, 'angle': angle_degrees, 'safety_score': 0.0}
        
        min_path_distance = min(path_distances)
        mean_path_distance = sum(path_distances) / len(path_distances)
        
        # Calculate density score (lower density = better path)
        # Count how many points are in danger/warning zones
        danger_points = sum(1 for d in path_distances if d < self.analyzer.danger_distance)
        warning_points = sum(1 for d in path_distances if d < self.analyzer.warning_distance)
        safe_points = sum(1 for d in path_distances if d >= self.analyzer.safe_distance)
        
        total_points = len(path_distances)
        density_score = 1.0 - (danger_points * 1.0 + warning_points * 0.5) / total_points
        safe_ratio = safe_points / total_points
        
        # Calculate safety metrics - prioritize less dense areas
        is_safe = min_path_distance >= self.min_forward_distance and safe_ratio >= 0.6
        
        # Enhanced safety score that considers obstacle density
        distance_score = min(1.0, min_path_distance / self.analyzer.safe_distance)
        mean_distance_score = min(1.0, mean_path_distance / self.analyzer.safe_distance)
        
        safety_score = (distance_score * 0.4 + 
                       mean_distance_score * 0.3 + 
                       density_score * 0.2 + 
                       safe_ratio * 0.1)
        
        return {
            'is_safe': is_safe,
            'angle': angle_degrees,
            'min_distance': min_path_distance,
            'mean_distance': mean_path_distance,
            'safety_score': safety_score,
            'path_points': path_points,
            'density_score': density_score,
            'safe_ratio': safe_ratio
        }
    
    def _generate_maneuver_steps(self, best_path, rotation_steps):
        """Generate detailed maneuvering steps"""
        steps = []
        angle = best_path['angle']
        
        if abs(angle) <= 15:
            steps.append(f"Small adjustment: Rotate {abs(angle):.0f}° {'clockwise' if angle > 0 else 'counterclockwise'}")
            steps.append("Move forward when clear")
        elif abs(angle) <= 45:
            steps.append(f"Moderate turn: Rotate {abs(angle):.0f}° {'clockwise' if angle > 0 else 'counterclockwise'}")
            steps.append("Check clearance")
            steps.append("Move forward cautiously")
        else:
            steps.append(f"Major reorientation needed: {abs(angle):.0f}° turn")
            steps.append(f"Rotate {'clockwise' if angle > 0 else 'counterclockwise'} in {rotation_steps} steps")
            steps.append("Reassess after each 15° rotation")
            steps.append("Move forward when safe path confirmed")
        
        steps.append(f"Target path clearance: {best_path['min_distance']:.1f}m")
        
        return steps
    
    def get_detailed_guidance(self, assessment: NavigationAssessment) -> str:
        """Get detailed text guidance for the pilot"""
        if not assessment:
            return "No assessment available"
        
        guidance = []
        guidance.append(f"🎯 ACTION: {assessment.recommended_action.value.upper().replace('_', ' ')}")
        guidance.append(f"🔒 SAFETY: {assessment.risk_level} risk level")
        guidance.append(f"📏 OBSTACLE: {assessment.obstacle_distance:.1f}m ahead")
        
        if assessment.safe_path_angle is not None:
            guidance.append(f"🧭 SAFE PATH: {assessment.safe_path_angle:.0f}° {'right' if assessment.safe_path_angle > 0 else 'left'}")
        
        if assessment.rotation_direction:
            guidance.append(f"🔄 ROTATE: {assessment.rotation_direction}")
        
        guidance.append(f"✅ CONFIDENCE: {assessment.confidence:.0%}")
        
        return " | ".join(guidance)
    
    def draw_navigation_overlay(self, surface, distance_map, assessment: NavigationAssessment, x_offset=0, y_offset=0):
        """Draw navigation guidance overlay on the surface - centered on camera frame"""
        if assessment is None:
            return
        
        # Get surface dimensions (this should match the video frame size)
        surface_h, surface_w = surface.get_height(), surface.get_width()
        
        # Calculate center of the actual video frame
        center_x = surface_w // 2
        center_y = surface_h // 2
        
        # Generate and draw flight paths
        flight_paths = self.generate_flight_paths_for_display(surface_w, surface_h, assessment)
        
        # Draw flight paths with enhanced visibility
        for i, path in enumerate(flight_paths):
            if i == 0:  # Primary path
                color = (0, 255, 0)  # Bright green
                thickness = 6
                outline_color = (0, 0, 0)  # Black outline
                outline_thickness = 8
            else:  # Alternative paths
                color = (0, 255, 255)  # Cyan
                thickness = 4
                outline_color = (0, 0, 0)
                outline_thickness = 6
            
            # Draw path line with outline for better visibility
            if len(path.waypoints) >= 2:
                # Draw black outline first
                pygame.draw.lines(surface, outline_color, False, path.waypoints, outline_thickness)
                # Draw colored line on top
                pygame.draw.lines(surface, color, False, path.waypoints, thickness)
                
                # Draw direction arrow at end - larger and more visible
                if len(path.waypoints) >= 2:
                    end_point = path.waypoints[-1]
                    prev_point = path.waypoints[-2]
                    
                    # Calculate arrow direction
                    dx = end_point[0] - prev_point[0]
                    dy = end_point[1] - prev_point[1]
                    length = math.sqrt(dx*dx + dy*dy)
                    
                    if length > 0:
                        # Normalize direction
                        dx /= length
                        dy /= length
                        
                        # Draw larger arrowhead
                        arrow_length = 25
                        arrow_angle = math.pi / 5  # Slightly narrower arrow
                        
                        # Arrow points
                        p1_x = end_point[0] - arrow_length * (dx * math.cos(arrow_angle) - dy * math.sin(arrow_angle))
                        p1_y = end_point[1] - arrow_length * (dx * math.sin(arrow_angle) + dy * math.cos(arrow_angle))
                        p2_x = end_point[0] - arrow_length * (dx * math.cos(-arrow_angle) - dy * math.sin(-arrow_angle))
                        p2_y = end_point[1] - arrow_length * (dx * math.sin(-arrow_angle) + dy * math.cos(-arrow_angle))
                        
                        # Draw arrow outline
                        pygame.draw.polygon(surface, outline_color, 
                                          [end_point, (int(p1_x)-1, int(p1_y)-1), (int(p2_x)-1, int(p2_y)-1)])
                        # Draw arrow
                        pygame.draw.polygon(surface, color, 
                                          [end_point, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y))])
        
        # Draw current position indicator (drone center) - larger and more visible
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), 16)  # Black outline
        pygame.draw.circle(surface, (255, 255, 0), (center_x, center_y), 14)  # Yellow center
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), 14, 3)   # Black border
        
        # Draw crosshair for center reference - more visible
        cross_size = 25
        pygame.draw.line(surface, (0, 0, 0), 
                        (center_x - cross_size, center_y), 
                        (center_x + cross_size, center_y), 5)  # Black outline
        pygame.draw.line(surface, (255, 255, 255), 
                        (center_x - cross_size, center_y), 
                        (center_x + cross_size, center_y), 3)  # White line
        pygame.draw.line(surface, (0, 0, 0), 
                        (center_x, center_y - cross_size), 
                        (center_x, center_y + cross_size), 5)  # Black outline
        pygame.draw.line(surface, (255, 255, 255), 
                        (center_x, center_y - cross_size), 
                        (center_x, center_y + cross_size), 3)  # White line
        
        # Draw obstacle warning zones if present
        if not assessment.is_safe_to_proceed and assessment.obstacle_distance < 2.0:
            # Draw pulsing warning circle
            warning_radius = int(60 + 20 * math.sin(time.time() * 3))  # Slower, smaller pulse
            pygame.draw.circle(surface, (255, 0, 0), (center_x, center_y), warning_radius, 4)
            
        # Add path confidence indicator
        if flight_paths:
            confidence = assessment.confidence if assessment.confidence else 0.0
            confidence_text = f"Path Confidence: {confidence:.0%}"
            
            # Create text surface with outline for visibility - with error handling
            try:
                text_outline = self.small_font.render(confidence_text, True, (0, 0, 0))
                text_main = self.small_font.render(confidence_text, True, (255, 255, 255))
                
                # Position at top of video
                text_x = center_x - text_main.get_width() // 2
                text_y = 10
                
                # Draw text with outline
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            surface.blit(text_outline, (text_x + dx, text_y + dy))
                surface.blit(text_main, (text_x, text_y))
            except Exception as e:
                print(f"Warning: Could not render confidence text: {e}")
    
    def generate_flight_paths_for_display(self, surface_w, surface_h, assessment: NavigationAssessment) -> List[FlightPath]:
        """Generate flight paths optimized for display on camera frame"""
        center_x = surface_w // 2
        center_y = surface_h // 2
        
        flight_paths = []
        
        # Always generate a path if we have an assessment
        if assessment:
            safe_angle = assessment.safe_path_angle if assessment.safe_path_angle is not None else 0.0
            
            # Generate primary recommended path pointing towards less dense area
            angle_rad = math.radians(safe_angle)
            
            # Calculate path length based on surface size - make it longer and more visible
            path_length = min(surface_w, surface_h) // 2.5  # Increased length
            
            # Direct path from center
            end_x = center_x + int(path_length * math.sin(angle_rad))
            end_y = center_y - int(path_length * math.cos(angle_rad))  # Negative because screen y increases downward
            
            # Ensure end point is within bounds
            end_x = max(30, min(surface_w - 30, end_x))
            end_y = max(30, min(surface_h - 30, end_y))
            
            # Create intermediate points for smoother visualization
            intermediate_x = center_x + int((path_length * 0.6) * math.sin(angle_rad))
            intermediate_y = center_y - int((path_length * 0.6) * math.cos(angle_rad))
            
            direct_path = FlightPath(
                start_point=(center_x, center_y),
                end_point=(end_x, end_y),
                waypoints=[(center_x, center_y), (intermediate_x, intermediate_y), (end_x, end_y)],
                safety_score=assessment.confidence if assessment.confidence else 0.5,
                path_type="direct"
            )
            flight_paths.append(direct_path)
            
            # Generate alternative path if we have multiple options
            if assessment.alternative_paths and len(assessment.alternative_paths) > 0:
                # Show best alternative path
                alt_path = assessment.alternative_paths[0]
                if 'angle' in alt_path:
                    alt_angle_rad = math.radians(alt_path['angle'])
                    alt_end_x = center_x + int((path_length * 0.8) * math.sin(alt_angle_rad))
                    alt_end_y = center_y - int((path_length * 0.8) * math.cos(alt_angle_rad))
                    
                    # Ensure within bounds
                    alt_end_x = max(30, min(surface_w - 30, alt_end_x))
                    alt_end_y = max(30, min(surface_h - 30, alt_end_y))
                    
                    alt_path_obj = FlightPath(
                        start_point=(center_x, center_y),
                        end_point=(alt_end_x, alt_end_y),
                        waypoints=[(center_x, center_y), (alt_end_x, alt_end_y)],
                        safety_score=alt_path.get('safety_score', 0.3),
                        path_type="alternative"
                    )
                    flight_paths.append(alt_path_obj)
        
        return flight_paths

def main():
    """Main function to run the real-time drone navigation system"""
    print("🚁 Real-Time Drone Navigation System")
    print("="*50)
    print("🎯 Analyzing camera feed with obstacle avoidance guidance")
    print("📷 Press SPACE to save analysis data")
    print("🔴 Press ESC to quit")
    print("="*50)
    
    # Create analyzer
    analyzer = RealTimeDroneAnalyzer()
    
    try:
        # Start analysis (camera index 1 is usually external webcam)
        analyzer.start_analysis(camera_index=0)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        analyzer.stop_analysis()

if __name__ == "__main__":
    main()