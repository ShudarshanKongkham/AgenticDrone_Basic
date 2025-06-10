"""
Complete Tello Drone Control and Sensor Monitoring System
=========================================================

This script provides comprehensive control for the Tello drone with real-time sensor monitoring.
Based on DJITelloPy documentation: https://djitellopy.readthedocs.io/en/latest/tello/

Features:
- All basic flight controls (takeoff, land, movement, rotation, flips)
- Complete sensor data access (gyroscope, accelerometer, barometer, temperature, etc.)
- Real-time video streaming with sensor overlay
- Comprehensive sensor data display and logging
- Safety features and error handling

Installation:
    pip install djitellopy opencv-python numpy

Usage:
    python tello_control.py
"""

import cv2
import time
import numpy as np
from djitellopy import Tello
from datetime import datetime
import json
from typing import Dict, Any

class TelloController:
    """Complete Tello drone controller with comprehensive sensor access."""
    
    def __init__(self):
        try:
            self.drone = Tello()
            self.connected = False
            self.flying = False
            self.video_on = False
            
            # Sensor caching to reduce command spam
            self.last_wifi_check = 0
            self.cached_wifi_snr = "N/A"
            self.wifi_check_interval = 5.0  # Check WiFi every 5 seconds
            
        except Exception as e:
            print(f"❌ Failed to create Tello instance: {e}")
            print("This usually means another Tello connection is active.")
            print("Please close any other Tello programs and try again.")
            raise
        
    def connect(self):
        """Connect to the Tello drone."""
        try:
            print("🔌 Connecting to Tello...")
            self.drone.connect()
            self.connected = True
            print("✅ Connected to Tello!")
            
            # Try to get basic info, but don't fail if sensors aren't ready yet
            try:
                battery = self.drone.get_battery()
                print(f"🔋 Battery: {battery}%")
            except:
                print("📊 Sensor data will be available after initialization...")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Make sure Tello is powered on and connected to WiFi")
            return False
    
    def disconnect(self):
        """Safely disconnect from the drone."""
        try:
            if self.video_on:
                self.drone.streamoff()
            if self.flying:
                self.drone.land()
                time.sleep(3)
            
            self.drone.end()
            self.connected = False
            print("✅ Safely disconnected from Tello")
            
        except Exception as e:
            print(f"❌ Disconnect error: {e}")
    
    def print_drone_info(self):
        """Print initial drone information."""
        print("\n" + "="*50)
        print("🚁 TELLO DRONE INFORMATION")
        print("="*50)
        
        # Try each sensor individually
        try:
            sdk = self.drone.query_sdk_version()
            print(f"📝 SDK Version: {sdk}")
        except:
            print("📝 SDK Version: Not available")
        
        try:
            serial = self.drone.query_serial_number()
            print(f"🆔 Serial Number: {serial}")
        except:
            print("🆔 Serial Number: Not available")
        
        try:
            battery = self.drone.get_battery()
            print(f"🔋 Battery: {battery}%")
        except:
            print("🔋 Battery: Not available")
        
        try:
            temp = self.drone.get_temperature()
            print(f"🌡️ Temperature: {temp}°C")
        except:
            print("🌡️ Temperature: Not available")
        
        try:
            wifi = self.drone.query_wifi_signal_noise_ratio()
            print(f"📶 WiFi SNR: {wifi}")
        except:
            print("📶 WiFi SNR: Not available")
        
        print("="*50 + "\n")
    
    # ================= FLIGHT CONTROLS =================
    
    def takeoff(self):
        """Takeoff the drone."""
        if not self.connected:
            print("❌ Not connected to drone")
            return False
        
        try:
            print("🛫 Taking off...")
            self.drone.takeoff()
            self.flying = True
            print("✅ Takeoff successful")
            time.sleep(2)  # Allow stabilization
            return True
        except Exception as e:
            print(f"❌ Takeoff failed: {e}")
            return False
    
    def land(self):
        """Land the drone."""
        if not self.flying:
            print("❌ Drone is not flying")
            return False
        
        try:
            print("🛬 Landing...")
            
            # Stop video streaming first to reduce command load
            if self.video_on:
                print("📹 Stopping video before landing...")
                try:
                    self.drone.streamoff()
                    self.video_on = False
                    time.sleep(1)  # Give time for video to stop
                except:
                    pass  # Don't fail landing if video stop fails
            
            # Send land command with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.drone.land()
                    self.flying = False
                    print("✅ Landing successful")
                    time.sleep(2)  # Allow time for landing
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Landing attempt {attempt + 1} failed, retrying... ({e})")
                        time.sleep(1)
                    else:
                        raise e
                        
        except Exception as e:
            print(f"❌ Landing failed after {max_retries} attempts: {e}")
            print("💡 Try:")
            print("   - Use emergency stop if drone won't land")
            print("   - Check battery level")
            print("   - Restart drone if unresponsive")
            return False
    
    def emergency_stop(self):
        """Emergency stop - immediately cut motors."""
        try:
            print("🚨 EMERGENCY STOP!")
            self.drone.emergency()
            self.flying = False
        except Exception as e:
            print(f"❌ Emergency stop failed: {e}")
    
    # ================= MOVEMENT CONTROLS =================
    
    def move_up(self, distance=20):
        """Move up by specified distance (cm)."""
        try:
            self.drone.move_up(distance)
            print(f"⬆️ Moved up {distance}cm")
        except Exception as e:
            print(f"❌ Move up failed: {e}")
    
    def move_down(self, distance=20):
        """Move down by specified distance (cm)."""
        try:
            self.drone.move_down(distance)
            print(f"⬇️ Moved down {distance}cm")
        except Exception as e:
            print(f"❌ Move down failed: {e}")
    
    def move_left(self, distance=20):
        """Move left by specified distance (cm)."""
        try:
            self.drone.move_left(distance)
            print(f"⬅️ Moved left {distance}cm")
        except Exception as e:
            print(f"❌ Move left failed: {e}")
    
    def move_right(self, distance=20):
        """Move right by specified distance (cm)."""
        try:
            self.drone.move_right(distance)
            print(f"➡️ Moved right {distance}cm")
        except Exception as e:
            print(f"❌ Move right failed: {e}")
    
    def move_forward(self, distance=20):
        """Move forward by specified distance (cm)."""
        try:
            self.drone.move_forward(distance)
            print(f"🔼 Moved forward {distance}cm")
        except Exception as e:
            print(f"❌ Move forward failed: {e}")
    
    def move_back(self, distance=20):
        """Move backward by specified distance (cm)."""
        try:
            self.drone.move_back(distance)
            print(f"🔽 Moved back {distance}cm")
        except Exception as e:
            print(f"❌ Move back failed: {e}")
    
    # ================= ROTATION CONTROLS =================
    
    def rotate_clockwise(self, degrees=15):
        """Rotate clockwise by specified degrees."""
        try:
            self.drone.rotate_clockwise(degrees)
            print(f"🔄 Rotated clockwise {degrees}°")
        except Exception as e:
            print(f"❌ Clockwise rotation failed: {e}")
    
    def rotate_counter_clockwise(self, degrees=15):
        """Rotate counter-clockwise by specified degrees."""
        try:
            self.drone.rotate_counter_clockwise(degrees)
            print(f"🔄 Rotated counter-clockwise {degrees}°")
        except Exception as e:
            print(f"❌ Counter-clockwise rotation failed: {e}")
    
    # ================= FLIP MANEUVERS =================
    
    def flip_forward(self):
        """Perform forward flip."""
        try:
            self.drone.flip_forward()
            print("🤸 Forward flip completed")
        except Exception as e:
            print(f"❌ Forward flip failed: {e}")
    
    def flip_back(self):
        """Perform backward flip."""
        try:
            self.drone.flip_back()
            print("🤸 Backward flip completed")
        except Exception as e:
            print(f"❌ Backward flip failed: {e}")
    
    def flip_left(self):
        """Perform left flip."""
        try:
            self.drone.flip_left()
            print("🤸 Left flip completed")
        except Exception as e:
            print(f"❌ Left flip failed: {e}")
    
    def flip_right(self):
        """Perform right flip."""
        try:
            self.drone.flip_right()
            print("🤸 Right flip completed")
        except Exception as e:
            print(f"❌ Right flip failed: {e}")
    
    # ================= VIDEO STREAMING =================
    
    def start_video(self):
        """Start video streaming."""
        try:
            if self.video_on:
                print("📹 Video already streaming")
                return True
                
            print("📹 Starting video stream...")
            self.drone.streamon()
            
            # Wait a moment for stream to initialize
            time.sleep(2)
            
            # Test if we can get a frame
            try:
                frame_read = self.drone.get_frame_read()
                test_frame = frame_read.frame
                if test_frame is not None:
                    self.video_on = True
                    print("✅ Video stream started successfully")
                    return True
                else:
                    print("❌ Video stream started but no frames available")
                    return False
            except Exception as frame_error:
                print(f"❌ Video stream started but frame test failed: {frame_error}")
                return False
                
        except Exception as e:
            print(f"❌ Video start failed: {e}")
            print("💡 Common fixes:")
            print("   - Make sure drone is connected")
            print("   - Try restarting the drone")
            print("   - Check if another app is using the camera")
            return False
    
    def stop_video(self):
        """Stop video streaming."""
        try:
            if not self.video_on:
                print("📹 Video already stopped")
                return
                
            print("📹 Stopping video stream...")
            
            # Try to stop video with timeout and error handling
            try:
                # Some Tello firmwares respond with '90' instead of 'ok'
                # This is still considered successful
                self.drone.streamoff()
                print("✅ Video stream stopped")
            except Exception as e:
                # Check if it's just a response code issue
                error_str = str(e).lower()
                if "'90'" in error_str or "unsuccessful" in error_str:
                    print("✅ Video stream stopped (response code '90')")
                else:
                    raise e
                    
            self.video_on = False
            time.sleep(0.5)  # Give time for video to fully stop
            
        except Exception as e:
            print(f"❌ Video stop failed: {e}")
            # Force video off even if command failed
            self.video_on = False
    
    def get_frame(self):
        """Get current video frame."""
        if not self.video_on:
            return None
        try:
            frame_read = self.drone.get_frame_read()
            if frame_read is None:
                return None
                
            frame = frame_read.frame
            if frame is None:
                return None
                
            # Convert BGR to RGB for consistency
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"❌ Frame capture failed: {e}")
            return None
    
    # ================= COMPREHENSIVE SENSOR DATA =================
    
    def get_all_sensors(self) -> Dict[str, Any]:
        """Get comprehensive sensor data from all available sensors."""
        try:
            # Basic status and power
            battery = self.drone.get_battery()
            height = self.drone.get_height()
            temperature = self.drone.get_temperature()
            flight_time = self.drone.get_flight_time()
            
            # Barometer (altitude measurement in cm, not pressure!)
            # Note: 'baro' field represents altitude in cm, NOT atmospheric pressure
            baro_raw = self.drone.get_state_field('baro')
            barometer_altitude_cm = baro_raw * 100 if baro_raw else 0
            
            # Motion sensors - Speed (cm/s)
            speed_x = self.drone.get_speed_x()
            speed_y = self.drone.get_speed_y()
            speed_z = self.drone.get_speed_z()
            
            # Motion sensors - Acceleration (0.001g units, milli-g)
            accel_x = self.drone.get_acceleration_x()
            accel_y = self.drone.get_acceleration_y()
            accel_z = self.drone.get_acceleration_z()
            
            # Attitude/Gyroscope (degrees)
            pitch = self.drone.get_pitch()
            roll = self.drone.get_roll()
            yaw = self.drone.get_yaw()
            
            # Distance sensors
            distance_tof = self.drone.get_distance_tof()
            
            # Temperature sensors
            temp_high = self.drone.get_highest_temperature()
            temp_low = self.drone.get_lowest_temperature()
            
            # Network and system - Use cached WiFi to reduce command spam
            current_time = time.time()
            if current_time - self.last_wifi_check > self.wifi_check_interval:
                try:
                    self.cached_wifi_snr = self.drone.query_wifi_signal_noise_ratio()
                    self.last_wifi_check = current_time
                except Exception:
                    # Don't fail sensor reading if WiFi query fails
                    pass
            wifi_snr = self.cached_wifi_snr
            
            # Calculate derived values
            total_speed = np.sqrt(speed_x**2 + speed_y**2 + speed_z**2)
            total_acceleration = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            
            sensor_data = {
                # Power and status
                'battery_percent': battery,
                'flight_time_seconds': flight_time,
                'height_cm': height,
                
                # Environmental sensors
                'temperature_avg_celsius': temperature,
                'temperature_high_celsius': temp_high,
                'temperature_low_celsius': temp_low,
                'barometer_altitude_cm': barometer_altitude_cm,
                
                # Motion - Linear velocity (cm/s)
                'velocity_x_cms': speed_x,
                'velocity_y_cms': speed_y,
                'velocity_z_cms': speed_z,
                'velocity_total_cms': total_speed,
                
                # Motion - Acceleration (milli-g, 0.001g units)
                'acceleration_x_mg': accel_x,
                'acceleration_y_mg': accel_y,
                'acceleration_z_mg': accel_z,
                'acceleration_total_mg': total_acceleration,
                
                # Attitude/Gyroscope (degrees)
                'pitch_degrees': pitch,
                'roll_degrees': roll,
                'yaw_degrees': yaw,
                
                # Distance sensors
                'distance_tof_cm': distance_tof,
                
                # Network
                'wifi_signal_noise_ratio': wifi_snr,
                
                # Status
                'is_connected': self.connected,
                'is_flying': self.flying,
                'video_streaming': self.video_on,
                
                # Timestamp
                'timestamp': time.time(),
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            }
            
            return sensor_data
            
        except Exception as e:
            print(f"❌ Sensor reading error: {e}")
            return {}
    
    def display_sensors(self, sensor_data: Dict[str, Any]):
        """Display sensor data in a formatted console output."""
        if not sensor_data:
            return
        
        # Clear screen
        print("\033[2J\033[H", end="")
        
        print("🚁" + "="*80 + "🚁")
        print(f"                    TELLO COMPREHENSIVE SENSOR DATA")
        print(f"                         {sensor_data['datetime']}")
        print("🚁" + "="*80 + "🚁")
        
        # Status Section
        print("\n📊 DRONE STATUS:")
        print(f"   📶 Connected: {'🟢 YES' if sensor_data['is_connected'] else '🔴 NO'}")
        print(f"   🛫 Flying: {'🟢 YES' if sensor_data['is_flying'] else '🔴 NO'}")
        print(f"   📹 Video: {'🟢 STREAMING' if sensor_data['video_streaming'] else '🔴 OFF'}")
        
        # Power and Environment
        print(f"\n🔋 POWER & ENVIRONMENT:")
        battery = sensor_data['battery_percent']
        bat_bar = "█" * (battery // 5) + "░" * (20 - battery // 5)
        print(f"   🔋 Battery: {battery:3d}% [{bat_bar}]")
        print(f"   ⏱️ Flight Time: {sensor_data['flight_time_seconds']:4d} seconds")
        print(f"   📏 Height: {sensor_data['height_cm']:4d} cm")
        print(f"   🌡️ Temperature: {sensor_data['temperature_avg_celsius']:5.1f}°C")
        print(f"   🌡️ Temp Range: {sensor_data['temperature_low_celsius']:.1f}°C - {sensor_data['temperature_high_celsius']:.1f}°C")
        print(f"   📏 Barometer Altitude: {sensor_data['barometer_altitude_cm']:7.1f} cm")
        print(f"   📶 WiFi SNR: {sensor_data['wifi_signal_noise_ratio']}")
        
        # Motion - Gyroscope/Attitude
        print(f"\n🎯 ATTITUDE/GYROSCOPE:")
        pitch = sensor_data['pitch_degrees']
        roll = sensor_data['roll_degrees']  
        yaw = sensor_data['yaw_degrees']
        print(f"   🔄 Pitch: {pitch:+4d}° (nose up/down)")
        print(f"   🔄 Roll:  {roll:+4d}° (left/right tilt)")
        print(f"   🔄 Yaw:   {yaw:+4d}° (rotation)")
        
        # Motion - Velocity
        print(f"\n🏃 VELOCITY (cm/s):")
        print(f"   ➡️ X-axis: {sensor_data['velocity_x_cms']:+7.1f} cm/s (left/right)")
        print(f"   ⬆️ Y-axis: {sensor_data['velocity_y_cms']:+7.1f} cm/s (forward/back)")
        print(f"   🔺 Z-axis: {sensor_data['velocity_z_cms']:+7.1f} cm/s (up/down)")
        print(f"   🚀 Total:  {sensor_data['velocity_total_cms']:7.1f} cm/s")
        
        # Motion - Acceleration  
        print(f"\n⚡ ACCELERATION (milli-g):")
        print(f"   ➡️ X-axis: {sensor_data['acceleration_x_mg']:+7.3f} mg")
        print(f"   ⬆️ Y-axis: {sensor_data['acceleration_y_mg']:+7.3f} mg")
        print(f"   🔺 Z-axis: {sensor_data['acceleration_z_mg']:+7.3f} mg")
        print(f"   ⚡ Total:  {sensor_data['acceleration_total_mg']:7.3f} mg")
        
        # Distance Sensors
        print(f"\n📡 DISTANCE SENSORS:")
        print(f"   📡 ToF Sensor: {sensor_data['distance_tof_cm']:4d} cm (downward facing)")
        
        print("\n" + "="*84)
        print("🎮 CONTROLS: Press 'q' to quit | 's' to save data | 'c' to clear screen")
        print("="*84)
    
    def save_sensor_data(self, sensor_data: Dict[str, Any], filename: str = None):
        """Save sensor data to JSON file."""
        if not sensor_data:
            print("❌ No sensor data to save")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tello_sensors_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(sensor_data, f, indent=2)
            print(f"💾 Sensor data saved to {filename}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def monitor_sensors_live(self, duration=60):
        """Monitor all sensors in real-time."""
        print(f"🚀 Starting live sensor monitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        data_history = []
        
        try:
            while time.time() - start_time < duration:
                sensor_data = self.get_all_sensors()
                if sensor_data:
                    data_history.append(sensor_data)
                    self.display_sensors(sensor_data)
                time.sleep(0.5)  # Update every 500ms
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        # Save collected data
        if data_history:
            save_data = input(f"\nSave {len(data_history)} sensor readings? (y/N): ")
            if save_data.lower() == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tello_sensor_history_{timestamp}.json"
                try:
                    with open(filename, 'w') as f:
                        json.dump(data_history, f, indent=2)
                    print(f"💾 Sensor history saved to {filename}")
                except Exception as e:
                    print(f"❌ Save failed: {e}")
    
    def video_with_sensor_overlay(self):
        """Display video stream with real-time sensor overlay."""
        if not self.start_video():
            return
        
        print("📹 Video with sensor overlay. Press 'q' to quit.")
        
        try:
            while True:
                frame = self.get_frame()
                if frame is not None:
                    sensor_data = self.get_all_sensors()
                    
                    if sensor_data:
                        # Create overlay
                        overlay = frame.copy()
                        h, w = overlay.shape[:2]
                        
                        # Semi-transparent background
                        cv2.rectangle(overlay, (10, 10), (450, 300), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                        
                        # Add sensor text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        color = (0, 255, 0)
                        thickness = 1
                        
                        sensor_text = [
                            f"Battery: {sensor_data['battery_percent']}%",
                            f"Height: {sensor_data['height_cm']}cm",
                            f"Temperature: {sensor_data['temperature_avg_celsius']:.1f}C",
                            f"Barometer Alt: {sensor_data['barometer_altitude_cm']:.1f}cm",
                            f"Pitch: {sensor_data['pitch_degrees']}°",
                            f"Roll: {sensor_data['roll_degrees']}°",
                            f"Yaw: {sensor_data['yaw_degrees']}°",
                            f"Speed X: {sensor_data['velocity_x_cms']:.1f}cm/s",
                            f"Speed Y: {sensor_data['velocity_y_cms']:.1f}cm/s",
                            f"Speed Z: {sensor_data['velocity_z_cms']:.1f}cm/s",
                            f"Total Speed: {sensor_data['velocity_total_cms']:.1f}cm/s",
                                        f"Accel X: {sensor_data['acceleration_x_mg']:.3f}mg",
            f"Accel Y: {sensor_data['acceleration_y_mg']:.3f}mg",
            f"Accel Z: {sensor_data['acceleration_z_mg']:.3f}mg",
                            f"ToF Distance: {sensor_data['distance_tof_cm']}cm"
                        ]
                        
                        for i, text in enumerate(sensor_text):
                            y_pos = 25 + i * 18
                            cv2.putText(frame, text, (20, y_pos), font, font_scale, color, thickness)
                    
                    cv2.imshow("Tello Video with Complete Sensor Data", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n🛑 Video stopped by user")
        finally:
            cv2.destroyAllWindows()
            self.stop_video()

def main():
    """Main demonstration function."""
    print("🚁" + "="*60 + "🚁")
    print("             TELLO COMPLETE CONTROL & SENSOR SYSTEM")
    print("🚁" + "="*60 + "🚁")
    print("\nThis system provides:")
    print("• Complete flight controls (takeoff, land, movement, rotation, flips)")
    print("• Comprehensive sensor monitoring (gyroscope, accelerometer, barometer, etc.)")
    print("• Real-time video with sensor overlay")
    print("• Data logging and export")
    print("\nMake sure your Tello is powered on and ready!\n")
    
    controller = TelloController()
    
    # Connect to drone
    if not controller.connect():
        print("❌ Cannot proceed without drone connection")
        return
    
    try:
        while True:
            print("\n🎮 TELLO CONTROL MENU:")
            print("1. Live sensor monitoring")
            print("2. Video with sensor overlay")
            print("3. Basic flight demo")
            print("4. Get single sensor reading")
            print("5. Advanced flight patterns")
            print("6. Flip demonstrations")
            print("0. Exit")
            
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                duration = input("Monitor duration in seconds (default 30): ").strip()
                duration = int(duration) if duration.isdigit() else 30
                controller.monitor_sensors_live(duration)
            elif choice == '2':
                controller.video_with_sensor_overlay()
            elif choice == '3':
                demo_basic_flight(controller)
            elif choice == '4':
                sensor_data = controller.get_all_sensors()
                controller.display_sensors(sensor_data)
                input("\nPress Enter to continue...")
            elif choice == '5':
                demo_advanced_flight(controller)
            elif choice == '6':
                demo_flips(controller)
            else:
                print("❌ Invalid option")
    
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
        if controller.flying:
            print("🚨 Emergency landing...")
            controller.emergency_stop()
    
    finally:
        controller.disconnect()
        print("✅ Program completed safely!")

def demo_basic_flight(controller):
    """Demonstrate basic flight operations."""
    print("\n✈️ BASIC FLIGHT DEMONSTRATION")
    print("⚠️ Ensure 3m x 3m clear space around drone!")
    
    battery = controller.drone.get_battery()
    if battery < 30:
        print(f"❌ Battery too low: {battery}% (need >30%)")
        return
    
    ready = input("Ready for flight demo? (y/N): ")
    if ready.lower() != 'y':
        return
    
    # Flight sequence
    if controller.takeoff():
        time.sleep(3)
        
        print("📊 Displaying flight sensor data...")
        sensor_data = controller.get_all_sensors()
        controller.display_sensors(sensor_data)
        time.sleep(2)
        
        # Basic movements
        print("🎮 Performing basic movements...")
        controller.move_up(50)
        time.sleep(2)
        controller.move_forward(50)
        time.sleep(2)
        controller.move_right(50)
        time.sleep(2)
        controller.move_back(50)
        time.sleep(2)
        controller.move_left(50)
        time.sleep(2)
        controller.move_down(50)
        time.sleep(2)
        
        # Rotations
        print("🔄 Performing rotations...")
        controller.rotate_clockwise(90)
        time.sleep(2)
        controller.rotate_counter_clockwise(180)
        time.sleep(2)
        controller.rotate_clockwise(90)
        time.sleep(2)
        
        controller.land()

def demo_advanced_flight(controller):
    """Demonstrate advanced flight patterns."""
    print("\n🚀 ADVANCED FLIGHT PATTERNS")
    
    battery = controller.drone.get_battery()
    if battery < 50:
        print(f"❌ Battery too low: {battery}% (need >50%)")
        return
    
    ready = input("Ready for advanced flight demo? (y/N): ")
    if ready.lower() != 'y':
        return
    
    if controller.takeoff():
        time.sleep(3)
        
        # Advanced movements using go_xyz_speed
        print("📍 Advanced positioning...")
        try:
            # Move in a square pattern
            controller.drone.go_xyz_speed(50, 0, 20, 30)  # Right and up
            time.sleep(3)
            controller.drone.go_xyz_speed(0, 50, 0, 30)   # Forward
            time.sleep(3)
            controller.drone.go_xyz_speed(-50, 0, 0, 30)  # Left
            time.sleep(3)
            controller.drone.go_xyz_speed(0, -50, -20, 30) # Back and down
            time.sleep(3)
        except Exception as e:
            print(f"❌ Advanced movement failed: {e}")
        
        controller.land()

def demo_flips(controller):
    """Demonstrate flip maneuvers."""
    print("\n🤸 FLIP DEMONSTRATIONS")
    print("⚠️ WARNING: Flips require lots of space and >50% battery!")
    
    battery = controller.drone.get_battery()
    if battery < 50:
        print(f"❌ Battery too low: {battery}% (need >50%)")
        return
    
    ready = input("Ready for flip demo? (y/N): ")
    if ready.lower() != 'y':
        return
    
    if controller.takeoff():
        time.sleep(3)
        
        # Move to safe height for flips
        print("⬆️ Moving to safe altitude...")
        controller.move_up(100)
        time.sleep(3)
        
        # Perform flips
        flips = [
            ("Forward flip", controller.flip_forward),
            ("Backward flip", controller.flip_back),
            ("Left flip", controller.flip_left),
            ("Right flip", controller.flip_right)
        ]
        
        for flip_name, flip_func in flips:
            print(f"🤸 Performing {flip_name}...")
            flip_func()
            time.sleep(4)  # Wait for flip completion
            
            # Show post-flip status
            sensor_data = controller.get_all_sensors()
            print(f"   📊 Post-flip: Battery {sensor_data['battery_percent']}%, Height {sensor_data['height_cm']}cm")
        
        controller.land()

if __name__ == "__main__":
    main()
