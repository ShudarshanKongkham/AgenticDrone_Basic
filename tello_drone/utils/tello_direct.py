"""
Direct Tello Control
===================

Direct UDP communication with Tello drone, bypassing DJITelloPy issues.
"""

import socket
import threading
import time
import cv2
import numpy as np

class TelloDirect:
    """Direct UDP communication with Tello."""
    
    def __init__(self):
        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.local_port = 9000
        
        self.socket = None
        self.connected = False
        self.response = ""
        self.state_data = {}
        
        # Video stream
        self.video_socket = None
        self.video_port = 11111
        self.video_active = False
        
    def connect(self):
        """Connect to Tello using direct UDP."""
        try:
            print("🔌 Creating UDP socket...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('', self.local_port))
            
            print("🔌 Sending command...")
            response = self.send_command('command')
            
            if response == 'ok':
                self.connected = True
                print("✅ Connected successfully!")
                
                # Get basic info
                try:
                    battery = self.send_command('battery?')
                    print(f"🔋 Battery: {battery}%")
                except:
                    print("⚠️ Could not get battery info")
                
                return True
            else:
                print(f"❌ Connection failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def send_command(self, command, timeout=5):
        """Send command and wait for response."""
        if not self.socket:
            return "No socket"
        
        try:
            # Send command
            self.socket.sendto(command.encode(), (self.tello_ip, self.tello_port))
            
            # Wait for response
            self.socket.settimeout(timeout)
            data, addr = self.socket.recvfrom(1024)
            response = data.decode().strip()
            
            return response
            
        except socket.timeout:
            return "timeout"
        except Exception as e:
            return f"error: {e}"
    
    def takeoff(self):
        """Takeoff command."""
        if not self.connected:
            return False
        
        response = self.send_command('takeoff')
        if response == 'ok':
            print("✅ Takeoff successful")
            return True
        else:
            print(f"❌ Takeoff failed: {response}")
            return False
    
    def land(self):
        """Land command."""
        if not self.connected:
            return False
        
        response = self.send_command('land')
        if response == 'ok':
            print("✅ Landing successful")
            return True
        else:
            print(f"❌ Landing failed: {response}")
            return False
    
    def move_up(self, distance):
        """Move up command."""
        response = self.send_command(f'up {distance}')
        return response == 'ok'
    
    def move_down(self, distance):
        """Move down command."""
        response = self.send_command(f'down {distance}')
        return response == 'ok'
    
    def move_left(self, distance):
        """Move left command."""
        response = self.send_command(f'left {distance}')
        return response == 'ok'
    
    def move_right(self, distance):
        """Move right command."""
        response = self.send_command(f'right {distance}')
        return response == 'ok'
    
    def move_forward(self, distance):
        """Move forward command."""
        response = self.send_command(f'forward {distance}')
        return response == 'ok'
    
    def move_back(self, distance):
        """Move backward command."""
        response = self.send_command(f'back {distance}')
        return response == 'ok'
    
    def rotate_clockwise(self, degrees):
        """Rotate clockwise."""
        response = self.send_command(f'cw {degrees}')
        return response == 'ok'
    
    def rotate_counter_clockwise(self, degrees):
        """Rotate counter-clockwise."""
        response = self.send_command(f'ccw {degrees}')
        return response == 'ok'
    
    def get_battery(self):
        """Get battery level."""
        try:
            response = self.send_command('battery?')
            return int(response)
        except:
            return 0
    
    def get_height(self):
        """Get current height."""
        try:
            response = self.send_command('height?')
            return int(response.replace('dm', '')) * 10  # Convert dm to cm
        except:
            return 0
    
    def get_temperature(self):
        """Get temperature."""
        try:
            response = self.send_command('temp?')
            # Response format: "60~62"
            temps = response.split('~')
            return (int(temps[0]) + int(temps[1])) / 2
        except:
            return 0
    
    def start_video(self):
        """Start video stream."""
        response = self.send_command('streamon')
        if response == 'ok':
            self.video_active = True
            print("📹 Video stream started")
            return True
        else:
            print(f"❌ Video failed: {response}")
            return False
    
    def stop_video(self):
        """Stop video stream."""
        response = self.send_command('streamoff')
        if response == 'ok':
            self.video_active = False
            print("📹 Video stream stopped")
            return True
        return False
    
    def get_frame_udp(self):
        """Get video frame via UDP (simplified)."""
        # Note: This is a simplified version
        # Full video implementation would need more complex UDP handling
        print("📹 Video frame capture not implemented in direct mode")
        print("Use DJITelloPy for video once connection issues are resolved")
        return None
    
    def emergency(self):
        """Emergency stop."""
        response = self.send_command('emergency')
        print("🚨 Emergency command sent")
        return response
    
    def disconnect(self):
        """Disconnect from Tello."""
        if self.video_active:
            self.stop_video()
        
        if self.socket:
            self.socket.close()
        
        self.connected = False
        print("✅ Disconnected")

def test_direct_control():
    """Test the direct control implementation."""
    print("🚁 Testing Direct Tello Control")
    print("="*40)
    
    tello = TelloDirect()
    
    if not tello.connect():
        print("❌ Connection failed")
        return
    
    try:
        # Get sensor data
        print(f"🔋 Battery: {tello.get_battery()}%")
        print(f"📏 Height: {tello.get_height()}cm")
        print(f"🌡️ Temperature: {tello.get_temperature()}°C")
        
        # Test flight controls
        flight_test = input("\n✈️ Test flight controls? (y/N): ")
        if flight_test.lower() == 'y':
            print("🛫 Taking off...")
            if tello.takeoff():
                time.sleep(3)
                
                print("⬆️ Moving up...")
                tello.move_up(30)
                time.sleep(2)
                
                print("🔄 Rotating...")
                tello.rotate_clockwise(45)
                time.sleep(2)
                
                print("⬇️ Moving down...")
                tello.move_down(30)
                time.sleep(2)
                
                print("🛬 Landing...")
                tello.land()
        
        # Test video
        video_test = input("\n📹 Test video stream? (y/N): ")
        if video_test.lower() == 'y':
            if tello.start_video():
                print("📹 Video started (frame capture not implemented)")
                time.sleep(3)
                tello.stop_video()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        tello.emergency()
    
    finally:
        tello.disconnect()

if __name__ == "__main__":
    test_direct_control() 