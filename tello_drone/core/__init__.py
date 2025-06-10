"""
Core Tello Drone Control Module
===============================

Contains the main TelloController class for:
- Flight control (takeoff, land, movement, rotation)
- Comprehensive sensor data access
- Video streaming capabilities
- Safety features and error handling
"""

from .tello_control import TelloController

__all__ = ['TelloController'] 