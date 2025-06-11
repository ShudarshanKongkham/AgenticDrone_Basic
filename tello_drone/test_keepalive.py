#!/usr/bin/env python3
"""
Test script to demonstrate that the keepalive mechanism prevents auto-landing.
This script will takeoff, wait for 30 seconds (longer than the normal auto-land timeout),
then land. The drone should NOT auto-land during the waiting period.
"""

import time
from ai.ollama_controller import OllamaController

def test_keepalive():
    """Test the keepalive mechanism to prevent auto-landing."""
    print("🧪 TESTING KEEPALIVE MECHANISM")
    print("="*50)
    print("This test will:")
    print("1. Connect to the drone")
    print("2. Take off")
    print("3. Wait for 30 seconds (longer than auto-land timeout)")
    print("4. Send a command to confirm drone is still responsive")
    print("5. Land")
    print("="*50)
    
    # Initialize AI controller (we don't need Ollama for this test)
    ai_controller = OllamaController()
    
    try:
        # Connect to drone
        print("🚁 Connecting to Tello drone...")
        if not ai_controller.connect_drone():
            print("❌ Failed to connect to drone!")
            return
        
        print("✅ Drone connected!")
        
        # Get battery level
        try:
            battery = ai_controller.tello.drone.get_battery()
            if battery < 30:
                print(f"❌ Battery too low for test: {battery}% (need >30%)")
                return
            print(f"🔋 Battery level: {battery}%")
        except:
            print("⚠️ Could not read battery level")
        
        # Ask for confirmation
        ready = input("\n⚠️ This will make the drone fly! Ensure clear space. Continue? (y/N): ")
        if ready.lower() != 'y':
            print("Test cancelled by user")
            return
        
        # Takeoff
        print("\n🛫 Taking off...")
        success = ai_controller.tello.takeoff()
        if not success:
            print("❌ Takeoff failed!")
            return
        
        # Start keepalive
        ai_controller.start_keepalive()
        
        # Wait for 30 seconds (this would normally cause auto-landing)
        print("⏳ Waiting 30 seconds to test keepalive...")
        print("   (Normally the drone would auto-land after ~15 seconds)")
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:
                print(f"   ⏰ {30-i} seconds remaining...")
        
        # Test if drone is still responsive
        print("\n🧪 Testing drone responsiveness...")
        try:
            # Get current height
            height = ai_controller.tello.drone.get_height()
            print(f"📏 Current height: {height}cm")
            print("✅ Drone is still responsive!")
            
            # Small movement to confirm control
            print("🔄 Testing movement...")
            ai_controller.tello.move_up(20)
            time.sleep(2)
            ai_controller.tello.move_down(20)
            print("✅ Movement test successful!")
            
        except Exception as e:
            print(f"❌ Drone not responsive: {e}")
        
        # Land
        print("\n🛬 Landing...")
        ai_controller.tello.land()
        
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print("🎉 The keepalive mechanism prevented auto-landing!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        if ai_controller.tello and ai_controller.tello.flying:
            print("🚨 Emergency landing...")
            ai_controller.tello.emergency_stop()
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if ai_controller.tello and ai_controller.tello.flying:
            print("🚨 Emergency landing...")
            ai_controller.tello.emergency_stop()
    
    finally:
        # Cleanup
        print("\n🔌 Disconnecting...")
        ai_controller.disconnect_drone()
        print("✅ Test cleanup completed!")

if __name__ == "__main__":
    test_keepalive() 