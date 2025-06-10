"""
Tello Drone GUI Module
======================

Contains the interactive HUD (Heads-Up Display) for:
- Real-time sensor monitoring
- Flight control interface
- Video streaming display
- Attitude indicators and visual feedback
- Data logging and export
"""

from .tello_hud import TelloHUD

__all__ = ['TelloHUD'] 