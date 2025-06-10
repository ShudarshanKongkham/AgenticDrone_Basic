#!/usr/bin/env python3
"""
Tello Drone HUD Launcher
========================

Launch the interactive Tello drone HUD application.

Usage:
    python launch_tello_hud.py

Requirements:
    - DJI Tello drone
    - WiFi connection to Tello network
    - Python packages: djitellopy, opencv-python, pillow, tkinter
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Launch the Tello HUD application."""
    try:
        from tello_drone.gui.tello_hud import TelloHUD
        
        print("🚁 Starting Tello Drone HUD...")
        print("Make sure your Tello is powered on and connected to WiFi")
        print("="*60)
        
        # Create and run the HUD
        hud = TelloHUD()
        hud.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Please install required packages:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Failed to start HUD: {e}")
        
    finally:
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 