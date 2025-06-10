#!/usr/bin/env python3
"""
Simple Tello Video Test Script
============================

This script tests basic video streaming functionality to help diagnose issues.

Usage:
    python test_video.py
"""

import cv2
import time
from djitellopy import Tello

def test_tello_video():
    """Test basic Tello video functionality."""
    print("🚁 Tello Video Test Script")
    print("=" * 40)
    
    try:
        # Create Tello instance
        print("1. Creating Tello instance...")
        drone = Tello()
        
        # Connect to drone
        print("2. Connecting to Tello...")
        drone.connect()
        print(f"✅ Connected! Battery: {drone.get_battery()}%")
        
        # Start video stream
        print("3. Starting video stream...")
        drone.streamon()
        print("✅ Video stream command sent")
        
        # Wait for video to initialize
        print("4. Waiting for video to initialize...")
        time.sleep(3)
        
        # Test frame capture
        print("5. Testing frame capture...")
        try:
            frame_read = drone.get_frame_read()
            frame = frame_read.frame
            
            if frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Frame captured successfully! Size: {width}x{height}")
                
                # Try to display frame for 5 seconds
                print("6. Displaying video for 5 seconds (press 'q' to quit early)...")
                start_time = time.time()
                
                while time.time() - start_time < 5:
                    frame = frame_read.frame
                    if frame is not None:
                        cv2.imshow('Tello Video Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("⚠️ Warning: Got empty frame")
                        
                cv2.destroyAllWindows()
                print("✅ Video test completed successfully!")
                
            else:
                print("❌ Failed to capture frame - frame is None")
                
        except Exception as frame_error:
            print(f"❌ Frame capture failed: {frame_error}")
            
        # Stop video stream
        print("7. Stopping video stream...")
        drone.streamoff()
        print("✅ Video stream stopped")
        
        # Disconnect
        print("8. Disconnecting...")
        drone.end()
        print("✅ Disconnected successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Common solutions:")
        print("   - Make sure Tello is powered on")
        print("   - Connect to Tello WiFi network")
        print("   - Close other apps using Tello")
        print("   - Restart Tello and try again")
        print("   - Check if OpenCV is installed: pip install opencv-python")

if __name__ == "__main__":
    test_tello_video() 