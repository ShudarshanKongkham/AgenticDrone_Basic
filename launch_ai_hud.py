#!/usr/bin/env python3
"""
Launch script for AI-Enhanced Tello HUD.
Combines manual controls with Ollama AI natural language processing.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tello_drone.gui.ai_hud import main

if __name__ == "__main__":
    print("🤖 Starting AI-Enhanced Tello HUD...")
    print("=" * 50)
    print("Features:")
    print("• Manual drone controls (left panel)")
    print("• Live video and sensor display (center)")  
    print("• AI natural language commands (right panel)")
    print("• Ollama integration for voice-like control")
    print("=" * 50)
    main() 