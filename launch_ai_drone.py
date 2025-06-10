#!/usr/bin/env python3
"""
Launch script for AI-powered Tello drone control using Ollama.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tello_drone.ai.ollama_controller import main

if __name__ == "__main__":
    main() 