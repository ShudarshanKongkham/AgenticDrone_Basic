# 🛡️ Ollama Drone Controller - Safety Features & Multi-Command Guide

## 🎯 Overview

The enhanced Ollama Controller now supports **multi-command sequences** with comprehensive **safety checks** to ensure safe drone operation while processing natural language instructions.

## 🚨 Safety Features

### 1. **Multi-Layer Safety Validation**

#### Distance & Movement Limits
- **Horizontal Movement**: 20-100cm (configurable up to 500cm)
- **Vertical Movement**: 20-200cm (configurable up to 500cm)  
- **Minimum Movement**: 20cm (prevents micro-movements)

#### Rotation Limits
- **Maximum Rotation**: 180° per command
- **Minimum Rotation**: 15° (prevents micro-rotations)
- **Safe Angles**: Any angle within limits

#### Flight State Validation
- **Flip Safety**: Only allowed while drone is flying
- **Emergency Priority**: Emergency commands always take precedence
- **Connection Check**: Validates drone connection before execution

### 2. **Safety Modes**

#### Normal Safety Mode (Default: ON)
```python
ai_controller.safety_mode = True  # Blocks unsafe commands
```

#### Advanced Safety Mode (Configurable)
```bash
🎯 Command: safety
# Interactive configuration menu
```

### 3. **Auto-Landing Prevention**
- **Keepalive System**: Sends commands every 5 seconds
- **Prevents**: Tello's 15-second auto-landing timeout
- **Auto-Start**: Begins after successful takeoff
- **Auto-Stop**: Ends when landing or emergency stop

## 🔄 Multi-Command Processing

### How It Works

1. **Natural Language Input**: "move forward 50cm and turn right 30 degrees"
2. **AI Parsing**: Ollama breaks down into individual commands
3. **Safety Validation**: Each command checked for safety
4. **Sequential Execution**: Commands executed with 2-second delays
5. **Error Handling**: User choice to continue on failures

### Example Multi-Commands

```bash
# Simple Sequences
"move forward 40cm and turn left 45 degrees"
"take off, move up 60cm, then rotate right"

# Complex Sequences  
"move left 50cm, turn around, then move forward 30cm"
"show status and move back 25cm"
"do a front flip then move back safely"

# Safety-First Examples
"take off, check status, then move up slowly"
"move forward carefully and land when done"
```

### Command Sequence Structure

```json
[
    {
        "action": "move_forward",
        "parameters": {"distance": 50},
        "sequence_order": 1,
        "safety_check": true,
        "explanation": "Move forward 50 centimeters"
    },
    {
        "action": "rotate_clockwise", 
        "parameters": {"degrees": 30},
        "sequence_order": 2,
        "safety_check": true,
        "explanation": "Turn right 30 degrees"
    }
]
```

## ⚙️ Configuration Options

### Safety Limits (Interactive)

```bash
🎯 Command: safety

🛡️ Safety Configuration
==============================
Current settings:
  • Safety mode: ON
  • Max movement distance: 100 cm
  • Max altitude change: 200 cm
==============================
Enable safety mode? (y/n): y
Max movement distance [current: 100cm]: 150
Max altitude change [current: 200cm]: 300
```

### Programmatic Configuration

```python
# Initialize with custom limits
ai_controller = OllamaController()
ai_controller.max_distance = 150      # 150cm max horizontal
ai_controller.max_altitude = 300      # 300cm max vertical
ai_controller.safety_mode = True      # Enable safety checks
```

## 🚨 Emergency Procedures

### Emergency Commands

| Command | Action | Safety Level |
|---------|--------|--------------|
| `emergency` | Emergency landing sequence | Critical |
| `emergency stop` | Immediate motor cut | Critical |
| `land now` | Safe landing | High |
| `abort` | Cancel current sequence | Medium |

### Emergency Landing Sequence

1. **Stop Keepalive**: Immediately stops auto-landing prevention
2. **Safe Landing**: Attempts controlled landing first
3. **Emergency Stop**: Falls back to motor cut if landing fails
4. **Status Report**: Confirms completion

```bash
🎯 Command: emergency

🚨 EMERGENCY LANDING SEQUENCE INITIATED
📡 Checking drone status...
🛬 Attempting safe landing...
✅ Safe landing completed
```

### Keyboard Interrupt Handling

```bash
# During command execution, Ctrl+C triggers:
🚨 Keyboard interrupt detected!
Initiating emergency landing sequence...
```

## 📊 Safety Validation Examples

### Valid Commands ✅

```bash
"move forward 50cm"           # Within limits
"turn right 45 degrees"       # Safe rotation  
"move up 30cm and hover"      # Safe sequence
"take off and show status"    # Logical sequence
```

### Blocked Commands ❌

```bash
"move forward 200cm"          # Exceeds distance limit
"rotate 300 degrees"          # Exceeds rotation limit
"move 5cm forward"            # Below minimum distance
"flip while landed"           # Not flying
```

### Safety Warnings ⚠️

```bash
❌ Command 1 failed safety check:
   - Movement 200cm exceeds limit 100cm
   - Rotation 300° too large (max 180°)
🛡️ Sequence blocked by safety mode
```

## 🧪 Testing

### Run Safety Tests

```bash
# Test parsing without drone
cd tello_drone/ai/
python test_multi_commands.py
```

### Test Output Example

```bash
🧪 Testing Multi-Command Parsing
==================================================
Test 1: 'move forward 50cm and turn right 30 degrees'
----------------------------------------
✅ Parsed 2 command(s):
  1. Action: move_forward
     Parameters: {'distance': 50}
     Safe: ✅
     Explanation: Move forward 50 centimeters
     
  2. Action: rotate_clockwise
     Parameters: {'degrees': 30}
     Safe: ✅
     Explanation: Turn right 30 degrees
```

## 🎮 Usage Examples

### Basic Multi-Command Usage

```python
from ollama_controller import OllamaController

# Initialize controller
ai_controller = OllamaController()

# Connect to drone
ai_controller.connect_drone()

# Process multi-command
success = ai_controller.process_natural_language_command(
    "move forward 40cm and turn left 45 degrees"
)

# Start interactive mode  
ai_controller.start_voice_mode()
```

### Interactive Session Example

```bash
🎯 Command: move forward 50cm and turn right 30 degrees

🤖 Processing multi-command: move forward 50cm and turn right 30 degrees
🎯 Executing command sequence (2 commands)
🎯 Executing sequence of 2 commands:

📋 Step 1: move_forward
   📝 Move forward 50 centimeters
   ✅ Step 1 completed

   ⏳ Waiting 2 seconds before next command...

📋 Step 2: rotate_clockwise  
   📝 Turn right 30 degrees
   ✅ Step 2 completed

📊 Sequence completed: 2/2 commands successful
✅ Command sequence completed successfully
```

## 🔧 Troubleshooting

### Common Issues

1. **"No valid commands found"**
   - Check Ollama connection
   - Verify model is installed
   - Try simpler commands

2. **"Sequence blocked by safety mode"**
   - Check distance/angle limits
   - Use `safety` command to adjust
   - Break into smaller commands

3. **"Command failed"**
   - Ensure drone is connected
   - Check drone battery level
   - Verify drone is in proper state

### Debug Mode

```python
# Enable detailed logging
ai_controller.debug = True
```

## 📚 Advanced Features

### Custom Safety Validators

```python
def custom_safety_check(action, parameters):
    # Add custom safety logic
    if action == "flip_forward" and battery_low():
        return False, ["Battery too low for flips"]
    return True, []

ai_controller.custom_validators.append(custom_safety_check)
```

### Command Logging

All commands are logged with timestamps:

```python
ai_controller.conversation_history
# [{"user": "move forward", "commands": ["move_forward"], "timestamp": "..."}]
```

## 🎯 Best Practices

1. **Start Simple**: Begin with single commands, then try sequences
2. **Test Safely**: Use test script before real drone flights
3. **Emergency Ready**: Know emergency commands by heart
4. **Stay Within Limits**: Respect safety boundaries
5. **Monitor Battery**: Check drone status regularly
6. **Safe Environment**: Fly in open areas away from obstacles

## 🤝 Contributing

To add new safety features:

1. Add validation in `validate_command_safety()`
2. Update multi-command parsing prompts
3. Add tests in `test_multi_commands.py`
4. Update this documentation

---

**⚠️ SAFETY DISCLAIMER**: Always test in controlled environments. Follow local drone regulations. The safety features are aids, not replacements for responsible piloting. 