# 🚁 Tello Drone Control System

A comprehensive Python package for controlling DJI Tello drones with an interactive GUI, real-time sensor monitoring, and video streaming capabilities.

![Tello Drone](https://img.shields.io/badge/Drone-DJI%20Tello-blue)
![Python](https://img.shields.io/badge/Python-3.7+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### 🤖 AI-Powered Control (NEW!)
- **Natural language commands** - "Take off and hover", "Move forward 50 cm", "Do a backflip"
- **Ollama integration** - Local LLM for intelligent command parsing
- **Safety validation** - AI checks commands for safety before execution
- **Conversational interface** - Chat-like interaction with your drone
- **Combined manual + AI control** - Switch between modes seamlessly

### 🎮 Interactive HUD (Heads-Up Display)
- **Real-time sensor monitoring** - Battery, temperature, altitude, gyroscope, acceleration
- **Interactive flight controls** - Takeoff, land, movement, rotation with visual buttons
- **Live video streaming** - 640x480 camera feed with toggle controls
- **Attitude indicator** - Artificial horizon with pitch, roll, yaw visualization
- **Data logging** - Export sensor data to JSON files

### 🛩️ Comprehensive Flight Control
- **Basic commands** - Takeoff, land, emergency stop
- **Movement** - Up/down, forward/back, left/right (20cm increments)
- **Rotation** - Clockwise/counter-clockwise (15° increments)
- **Advanced** - Flips, manual velocity control
- **Safety features** - Connection monitoring, error handling

### 📊 Complete Sensor Access
- **Power & Status** - Battery %, flight time, connection status
- **Motion Sensors** - Velocity (cm/s), acceleration (milli-g), gyroscope (degrees)
- **Environmental** - Temperature range, barometric altitude, WiFi signal
- **Distance** - Time-of-flight sensor, height measurement

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd Agentic_Drone

# Install dependencies
pip install -r requirements.txt
```

### 2. Connect to Tello
1. Power on your DJI Tello drone
2. Connect to the Tello WiFi network (usually "TELLO-XXXXXX")

### 3. Launch the Application

**Option A: AI-Enhanced HUD (🤖 Recommended)**
```bash
# Install Ollama first: https://ollama.ai
ollama pull llama3.1  # Download AI model
ollama serve          # Start Ollama server

# Launch AI HUD
python launch_ai_hud.py
```

**Option B: Classic HUD**
```bash
python launch_tello_hud.py
```

**Option C: Terminal AI Control**
```bash
python launch_ai_drone.py
```

### 4. Using the HUD
1. Click **🔌 Connect** to connect to the drone
2. Click **▶️ Start HUD** to begin sensor monitoring
3. Click **📹 Start Video** to enable camera streaming
4. Use flight control buttons for manual control
5. Monitor real-time data in the sensor panels

## 📁 Project Structure

```
Agentic_Drone/
├── 📦 tello_drone/              # Main package
│   ├── 🎯 core/                 # Core functionality
│   │   ├── __init__.py
│   │   └── tello_control.py     # Main controller class
│   ├── 🤖 ai/                   # AI integration (NEW!)
│   │   ├── __init__.py
│   │   └── ollama_controller.py # Natural language processing
│   ├── 🖥️ gui/                  # User interface
│   │   ├── __init__.py
│   │   ├── tello_hud.py         # Interactive HUD application
│   │   └── ai_hud.py            # AI-enhanced HUD (NEW!)
│   ├── 🔧 utils/                # Utilities and examples
│   │   ├── __init__.py
│   │   ├── tello_example.py     # Simple usage examples
│   │   └── tello_direct.py      # Direct UDP communication
│   └── 🧪 tests/                # Tests and diagnostics
│       ├── __init__.py
│       └── test_video.py        # Video streaming tests
├── 📚 documentation/            # Documentation
│   ├── api/                     # API documentation
│   ├── guides/                  # User guides
│   └── assets/                  # Images and assets
├── 📤 output/                   # Generated outputs
├── 📜 scripts/                  # Utility scripts
├── 🚀 launch_tello_hud.py      # Classic HUD launcher
├── 🤖 launch_ai_hud.py         # AI-enhanced HUD launcher (NEW!)
├── 🗣️ launch_ai_drone.py        # Terminal AI control launcher (NEW!)
├── 📋 requirements.txt          # Dependencies
└── 📖 README.md                # This file
```

## 🎯 Usage Examples

### AI Natural Language Control (NEW!)
```python
from tello_drone.ai.ollama_controller import OllamaController

# Create AI controller
ai = OllamaController()
ai.connect_drone()

# Natural language commands
ai.process_natural_language_command("Take off and hover")
ai.process_natural_language_command("Move forward 50 centimeters")  
ai.process_natural_language_command("Turn right 45 degrees")
ai.process_natural_language_command("Do a front flip")
ai.process_natural_language_command("Show me the sensor data")
ai.process_natural_language_command("Land safely")
```

### Interactive AI Chat Mode
```bash
python launch_ai_drone.py

# Then type commands like:
# "Take off"
# "Move up 30 cm and then forward 1 meter"
# "Turn around 180 degrees"
# "Show battery status"
# "Emergency land"
```

### Basic Control (Programmatic)
```python
from tello_drone.core.tello_control import TelloController

# Create controller and connect
controller = TelloController()
controller.connect()

# Basic flight
controller.takeoff()
controller.move_forward(50)  # Move 50cm forward
controller.rotate_clockwise(90)  # Turn 90 degrees
controller.land()

# Get sensor data
sensors = controller.get_all_sensors()
print(f"Battery: {sensors['battery_percent']}%")
print(f"Height: {sensors['height_cm']}cm")
```

### GUI Application
```python
from tello_drone.gui.tello_hud import TelloHUD

# Launch interactive HUD
hud = TelloHUD()
hud.run()
```

### Video Streaming Test
```python
# Test video functionality
python tello_drone/tests/test_video.py
```

## 🔧 Requirements

### Hardware
- **DJI Tello drone** (Tello, Tello EDU, or Tello Talent)
- **Computer with WiFi** capability
- **Windows/Mac/Linux** operating system

### Software
- **Python 3.7+**
- **Ollama** (for AI features) - Download from https://ollama.ai
- **Required packages** (see `requirements.txt`):
  - `djitellopy>=2.4.0` - Tello communication
  - `opencv-python>=4.5.0` - Video processing
  - `pillow>=8.0.0` - Image handling  
  - `numpy>=1.20.0` - Numerical operations
  - `requests>=2.25.0` - HTTP communication with Ollama
  - `tkinter` - GUI framework (usually included with Python)

## 🛠️ Configuration

### Video Settings
- **Default resolution**: 640x480 (resized from Tello's native resolution)
- **Frame rate**: 30 FPS
- **Display location**: Center panel below system info

### Movement Settings
- **Distance increments**: 20cm for movement commands
- **Rotation increments**: 15° for rotation commands
- **Safety features**: Connection validation, error handling

### Sensor Update Rate
- **HUD refresh**: 10 Hz (100ms intervals)
- **Video stream**: 30 FPS
- **Sensor polling**: Real-time via DJITelloPy

## 🐛 Troubleshooting

### Connection Issues
- **Ensure Tello is powered on** and status light is solid
- **Connect to Tello WiFi** network before running software
- **Close other applications** that might be using the Tello
- **Restart the drone** if connection fails

### Video Streaming Issues
- **Wait 3-5 seconds** after starting video for initialization
- **Check OpenCV installation**: `pip install opencv-python`
- **Try restarting** the drone if video fails
- **Use test script**: `python tello_drone/tests/test_video.py`

### Common Error Messages
- `"Did not receive a state packet"` → Check WiFi connection
- `"Video toggle error"` → Restart drone, check dependencies
- `"Import error"` → Install missing packages: `pip install -r requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DJI** for the Tello drone platform
- **DJITelloPy** library maintainers
- **OpenCV** community for computer vision tools
- **Python** community for excellent libraries

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the diagnostic test: `python tello_drone/tests/test_video.py`
3. Review the logs in the HUD activity panel
4. Open an issue with detailed error messages

---

**Happy Flying! 🚁✨** 