"""
Tello Drone Control Package
==========================

A comprehensive Python package for controlling DJI Tello drones with:
- Core flight control and sensor monitoring
- Interactive GUI with real-time HUD
- Utility functions and examples
- Video streaming capabilities

Main modules:
- core.tello_control: Core drone control and sensor access
- gui.tello_hud: Interactive GUI application
- utils: Helper utilities and examples
- tests: Test scripts and diagnostics

Author: Kong
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Kong"

# Import main classes for convenience
try:
    from .core.tello_control import TelloController
    from .gui.tello_hud import TelloHUD
except ImportError:
    # Handle case where dependencies aren't installed
    pass

__all__ = ['TelloController', 'TelloHUD'] 