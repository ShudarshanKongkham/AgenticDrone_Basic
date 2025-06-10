"""
Simple Tello Example Usage
=========================

This script shows basic usage of the TelloController class.
Based on your original code examples.
"""

import cv2
import time
from tello_control import TelloController

def basic_video_example():
    """Basic video streaming example similar to your original code."""
    print("🚁 Basic Video Streaming Example")
    print("="*40)
    
    # Create controller (equivalent to my_drone = Tello())
    controller = TelloController()
    
    # Connect (equivalent to my_drone.connect())
    if not controller.connect():
        print("❌ Failed to connect")
        return
    
    # Start video stream (equivalent to my_drone.streamon())
    if not controller.start_video():
        print("❌ Failed to start video")
        controller.disconnect()
        return
    
    print("📹 Video stream active. Press 'q' to quit.")
    
    try:
        while True:
            # Get frame (equivalent to my_drone.get_frame_read().frame)
            frame = controller.get_frame()
            if frame is not None:
                # Convert color (equivalent to cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Note: our get_frame() already does this conversion
                cv2.imshow("Tello Video", frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    finally:
        cv2.destroyAllWindows()
        controller.disconnect()

def sensor_data_example():
    """Example showing how to access sensor data."""
    print("\n📊 Sensor Data Example")
    print("="*30)
    
    controller = TelloController()
    
    if not controller.connect():
        return
    
    try:
        # Get all sensor data at once
        sensors = controller.get_all_sensors()
        
        print(f"🔋 Battery: {sensors['battery_percent']}%")
        print(f"📏 Height: {sensors['height_cm']} cm")
        print(f"🌡️ Temperature: {sensors['temperature_avg_celsius']}°C")
        
        # Access barometer like in your example: my_drone.get_state_field('baro') * 100
        # Note: This is altitude measurement in cm, not atmospheric pressure!
        print(f"📏 Barometer Altitude: {sensors['barometer_altitude_cm']} cm")
        
        # Access gyroscope data
        print(f"🎯 Pitch: {sensors['pitch_degrees']}°")
        print(f"🎯 Roll: {sensors['roll_degrees']}°")
        print(f"🎯 Yaw: {sensors['yaw_degrees']}°")
        
        # Access velocity data
        print(f"🏃 Speed X: {sensors['velocity_x_cms']} cm/s")
        print(f"🏃 Speed Y: {sensors['velocity_y_cms']} cm/s")
        print(f"🏃 Speed Z: {sensors['velocity_z_cms']} cm/s")
        
        # Access acceleration data
        print(f"⚡ Accel X: {sensors['acceleration_x_cms2']} cm/s²")
        print(f"⚡ Accel Y: {sensors['acceleration_y_cms2']} cm/s²")
        print(f"⚡ Accel Z: {sensors['acceleration_z_cms2']} cm/s²")
    
    finally:
        controller.disconnect()

def movement_control_example():
    """Example showing movement controls like in your original code."""
    print("\n🎮 Movement Control Example")
    print("="*35)
    
    controller = TelloController()
    
    if not controller.connect():
        return
    
    battery = controller.drone.get_battery()
    if battery < 30:
        print(f"❌ Battery too low for flight: {battery}%")
        controller.disconnect()
        return
    
    print("⚠️ This will make the drone fly! Ensure clear space.")
    ready = input("Continue? (y/N): ")
    
    if ready.lower() != 'y':
        controller.disconnect()
        return
    
    try:
        # Example similar to your signal-based control
        print('🛫 Drone take off')
        controller.takeoff()
        time.sleep(3)
        
        print('⬆️ Drone up')
        controller.move_up(20)
        time.sleep(2)
        
        print('⬇️ Drone down') 
        controller.move_down(20)
        time.sleep(2)
        
        print('⬅️ Drone left')
        controller.move_left(20)
        time.sleep(2)
        
        print('➡️ Drone right')
        controller.move_right(20)
        time.sleep(2)
        
        print('🔼 Drone forward')
        controller.move_forward(20)
        time.sleep(2)
        
        print('🔽 Drone backward')
        controller.move_back(20)
        time.sleep(2)
        
        print('🔄 Drone rotating clockwise')
        controller.rotate_clockwise(20)
        time.sleep(2)
        
        print('🔄 Drone rotating counter-clockwise')
        controller.rotate_counter_clockwise(20)
        time.sleep(2)
        
        print('🛬 Drone land')
        controller.land()
        time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n🚨 Emergency stop!")
        controller.emergency_stop()
    
    finally:
        controller.disconnect()

def main():
    """Main function with menu to choose examples."""
    print("🚁" + "="*50 + "🚁")
    print("           TELLO USAGE EXAMPLES")
    print("🚁" + "="*50 + "🚁")
    print("\nBased on your original code patterns:")
    print("1. Basic video streaming")
    print("2. Sensor data access")
    print("3. Movement controls")
    print("4. All examples")
    print("0. Exit")
    
    while True:
        choice = input("\nSelect example (0-4): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            basic_video_example()
        elif choice == '2':
            sensor_data_example()
        elif choice == '3':
            movement_control_example()
        elif choice == '4':
            basic_video_example()
            sensor_data_example()
            movement_control_example()
        else:
            print("❌ Invalid choice")
    
    print("✅ Examples completed!")

if __name__ == "__main__":
    main() 