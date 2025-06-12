#!/usr/bin/env python3
"""
Ollama AI Controller for Tello Drone
Integrates local LLM (Ollama) with Tello drone for natural language control.

Features:
- Natural language command processing using Ollama LLM
- Comprehensive drone control (takeoff, landing, movement, rotations, flips)
- Real-time sensor monitoring and status display
- Safety features and command validation
- Auto-landing prevention with keepalive mechanism
- Video streaming support
- Interactive voice/text control mode

The auto-landing prevention feature automatically sends keepalive commands
every 5 seconds while the drone is flying to prevent the Tello's built-in
15-second timeout that would cause automatic landing.
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

# Import our existing Tello controller
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.tello_control import TelloController


class OllamaController:
    """AI-powered drone controller using Ollama for natural language processing."""
    
    def __init__(self, ollama_host="http://localhost:11434", model="granite3.3"):
        """
        Initialize the Ollama controller.
        
        Args:
            ollama_host: Ollama server URL
            model: Model name to use (e.g., 'llama3.1', 'phi3', 'mistral')
        """
        self.ollama_host = ollama_host
        self.model = model
        self.tello = None
        self.conversation_history = []
        self.safety_mode = True
        self.max_altitude = 200  # cm
        self.max_distance = 100  # cm
        
        # Keepalive mechanism to prevent auto-landing
        self.keepalive_thread = None
        self.keepalive_running = False
        self.keepalive_interval = 5  # seconds (send keepalive every 5 seconds)
        
        # Command mapping for natural language processing
        self.command_patterns = {
            # Basic flight controls
            'takeoff': ['take off', 'takeoff', 'launch', 'fly up', 'start flying'],
            'land': ['land', 'landing', 'come down', 'touch down', 'stop flying'],
            'emergency': ['emergency', 'stop', 'emergency stop', 'cut motors', 'abort'],
            
            # Movement commands
            'move_up': ['up', 'ascend', 'climb', 'rise', 'go up', 'move up'],
            'move_down': ['down', 'descend', 'lower', 'drop', 'go down', 'move down'],
            'move_forward': ['forward', 'ahead', 'advance', 'go forward', 'move forward'],
            'move_back': ['back', 'backward', 'retreat', 'go back', 'move back'],
            'move_left': ['left', 'port', 'go left', 'move left'],
            'move_right': ['right', 'starboard', 'go right', 'move right'],
            
            # Rotation commands
            'rotate_clockwise': ['rotate right', 'turn right', 'clockwise', 'spin right'],
            'rotate_counter_clockwise': ['rotate left', 'turn left', 'counter clockwise', 'spin left'],
            
            # Flip commands
            'flip_forward': ['flip forward', 'front flip', 'forward flip'],
            'flip_back': ['flip back', 'back flip', 'backward flip'],
            'flip_left': ['flip left', 'left flip'],
            'flip_right': ['flip right', 'right flip'],
            
            # Video and sensors
            'start_video': ['start video', 'video on', 'camera on', 'start streaming'],
            'stop_video': ['stop video', 'video off', 'camera off', 'stop streaming'],
            'get_status': ['status', 'sensors', 'telemetry', 'info', 'data'],
        }
    
    def start_keepalive(self):
        """Start the keepalive thread to prevent auto-landing."""
        if self.keepalive_thread is None or not self.keepalive_thread.is_alive():
            self.keepalive_running = True
            self.keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
            self.keepalive_thread.start()
            print("🔄 Keepalive started - drone will not auto-land")
    
    def stop_keepalive(self):
        """Stop the keepalive thread."""
        self.keepalive_running = False
        if self.keepalive_thread and self.keepalive_thread.is_alive():
            self.keepalive_thread.join(timeout=1)
        print("⏹️ Keepalive stopped")
    
    def _keepalive_loop(self):
        """Background thread that sends keepalive commands to prevent auto-landing."""
        keepalive_count = 0
        while self.keepalive_running:
            try:
                if self.tello and self.tello.connected and self.tello.flying:
                    # Send keepalive command to prevent auto-landing
                    self.tello.drone.send_keepalive()
                    keepalive_count += 1
                    # Only print every 3rd keepalive (every 15 seconds) to reduce noise
                    if keepalive_count % 3 == 0:
                        print(f"💓 Keepalive active ({keepalive_count * self.keepalive_interval}s)")
                time.sleep(self.keepalive_interval)
            except Exception as e:
                print(f"⚠️ Keepalive error: {e}")
                time.sleep(self.keepalive_interval)
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception as e:
            print(f"❌ Failed to get models: {e}")
            return []
    
    def connect_drone(self) -> bool:
        """Connect to Tello drone."""
        try:
            print("🤖 Initializing AI-controlled Tello...")
            self.tello = TelloController()
            return self.tello.connect()
        except Exception as e:
            print(f"❌ Drone connection failed: {e}")
            return False
    
    def disconnect_drone(self):
        """Disconnect from Tello drone."""
        # Stop keepalive first
        self.stop_keepalive()
        
        if self.tello:
            self.tello.disconnect()
    
    def query_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Send query to Ollama and get response."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (last 5 exchanges)
            for entry in self.conversation_history[-10:]:
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["assistant"]})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent responses
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error querying Ollama: {e}"
    
    def parse_command(self, user_input: str) -> Dict[str, Any]:
        """Parse natural language input into drone commands using Ollama."""
        
        system_prompt = f"""You are an AI assistant that controls a Tello drone. Your job is to parse natural language commands into specific drone actions.

Available drone commands:
- takeoff: Take off and hover
- land: Land the drone safely
- emergency: Emergency stop (cuts motors immediately)
- move_up/move_down: Move vertically (20-100 cm)
- move_forward/move_back: Move horizontally forward/backward (20-100 cm)
- move_left/move_right: Move horizontally left/right (20-100 cm)
- rotate_clockwise/rotate_counter_clockwise: Rotate (15-90 degrees)
- flip_forward/flip_back/flip_left/flip_right: Perform flips
- start_video/stop_video: Control video streaming
- get_status: Get sensor data and drone status

Safety rules:
- Maximum movement distance: {self.max_distance} cm
- Maximum altitude movement: {self.max_altitude} cm
- Always confirm risky commands
- Refuse dangerous or impossible requests

Parse the user's command and respond with a JSON object containing:
{{
    "action": "command_name",
    "parameters": {{"distance": 50, "degrees": 30}},
    "safety_check": true/false,
    "explanation": "What the drone will do"
}}

If the command is unclear, ask for clarification.
If multiple actions are requested, choose the first/most important one.
If the command is dangerous, set safety_check to false and explain why.
"""

        user_prompt = f"Parse this drone command: '{user_input}'"
        
        print(f"🤖 Processing command: {user_input}")
        response = self.query_ollama(user_prompt, system_prompt)
        
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {
                    "action": "none",
                    "parameters": {},
                    "safety_check": False,
                    "explanation": response
                }
        except json.JSONDecodeError:
            return {
                "action": "none", 
                "parameters": {},
                "safety_check": False,
                "explanation": f"Could not parse command. AI response: {response}"
            }
    
    def execute_command(self, command_dict: Dict[str, Any]) -> bool:
        """Execute the parsed command on the drone."""
        if not self.tello or not self.tello.connected:
            print("❌ Drone not connected")
            return False
        
        action = command_dict.get('action', '')
        parameters = command_dict.get('parameters', {})
        safety_check = command_dict.get('safety_check', False)
        explanation = command_dict.get('explanation', '')
        
        print(f"🎯 Action: {action}")
        print(f"📝 Explanation: {explanation}")
        
        if not safety_check and self.safety_mode:
            print("⚠️ Safety check failed - command blocked")
            return False
        
        try:
            # Execute the command
            if action == 'takeoff':
                result = self.tello.takeoff()
                if result:
                    # Start keepalive after successful takeoff
                    self.start_keepalive()
                return result
            
            elif action == 'land':
                # Stop keepalive before landing
                self.stop_keepalive()
                return self.tello.land()
            
            elif action == 'emergency':
                # Stop keepalive on emergency
                self.stop_keepalive()
                self.tello.emergency_stop()
                return True
            
            elif action == 'move_up':
                distance = min(parameters.get('distance', 30), self.max_altitude)
                self.tello.move_up(distance)
                return True
            
            elif action == 'move_down':
                distance = min(parameters.get('distance', 30), self.max_altitude)
                self.tello.move_down(distance)
                return True
            
            elif action == 'move_forward':
                distance = min(parameters.get('distance', 30), self.max_distance)
                self.tello.move_forward(distance)
                return True
            
            elif action == 'move_back':
                distance = min(parameters.get('distance', 30), self.max_distance)
                self.tello.move_back(distance)
                return True
            
            elif action == 'move_left':
                distance = min(parameters.get('distance', 30), self.max_distance)
                self.tello.move_left(distance)
                return True
            
            elif action == 'move_right':
                distance = min(parameters.get('distance', 30), self.max_distance)
                self.tello.move_right(distance)
                return True
            
            elif action == 'rotate_clockwise':
                degrees = min(parameters.get('degrees', 30), 90)
                self.tello.rotate_clockwise(degrees)
                return True
            
            elif action == 'rotate_counter_clockwise':
                degrees = min(parameters.get('degrees', 30), 90)
                self.tello.rotate_counter_clockwise(degrees)
                return True
            
            elif action in ['flip_forward', 'flip_back', 'flip_left', 'flip_right']:
                if action == 'flip_forward':
                    self.tello.flip_forward()
                elif action == 'flip_back':
                    self.tello.flip_back()
                elif action == 'flip_left':
                    self.tello.flip_left()
                elif action == 'flip_right':
                    self.tello.flip_right()
                return True
            
            elif action == 'start_video':
                return self.tello.start_video()
            
            elif action == 'stop_video':
                self.tello.stop_video()
                return True
            
            elif action == 'get_status':
                sensors = self.tello.get_all_sensors()
                self.tello.display_sensors(sensors)
                return True
            
            else:
                print(f"❌ Unknown action: {action}")
                return False
                
        except Exception as e:
            print(f"❌ Command execution failed: {e}")
            return False
    
    def validate_command_safety(self, action: str, parameters: Dict) -> Dict[str, Any]:
        """Enhanced safety validation for commands."""
        safety_issues = []
        
        # Distance safety checks
        if action in ['move_up', 'move_down']:
            distance = parameters.get('distance', 30)
            if distance > self.max_altitude:
                safety_issues.append(f"Altitude movement {distance}cm exceeds limit {self.max_altitude}cm")
            elif distance < 20:
                safety_issues.append(f"Movement distance {distance}cm too small (min 20cm)")
        
        elif action in ['move_forward', 'move_back', 'move_left', 'move_right']:
            distance = parameters.get('distance', 30)
            if distance > self.max_distance:
                safety_issues.append(f"Movement {distance}cm exceeds limit {self.max_distance}cm")
            elif distance < 20:
                safety_issues.append(f"Movement distance {distance}cm too small (min 20cm)")
        
        # Rotation safety checks
        elif action in ['rotate_clockwise', 'rotate_counter_clockwise']:
            degrees = parameters.get('degrees', 30)
            if degrees > 180:
                safety_issues.append(f"Rotation {degrees}° too large (max 180°)")
            elif degrees < 15:
                safety_issues.append(f"Rotation {degrees}° too small (min 15°)")
        
        # Flip safety checks
        elif action in ['flip_forward', 'flip_back', 'flip_left', 'flip_right']:
            if not self.tello or not self.tello.flying:
                safety_issues.append("Flips can only be performed while flying")
        
        # Emergency and critical commands
        elif action == 'emergency':
            # Emergency is always allowed
            pass
        
        return {
            'safe': len(safety_issues) == 0,
            'issues': safety_issues,
            'action': action,
            'parameters': parameters
        }

    def parse_multi_commands(self, user_input: str) -> List[Dict[str, Any]]:
        """Parse natural language input that may contain multiple commands."""
        
        system_prompt = f"""You are an AI assistant that controls a Tello drone. Parse natural language commands into specific drone actions.

Available drone commands:
- takeoff: Take off and hover
- land: Land the drone safely
- emergency: Emergency stop (cuts motors immediately)
- move_up/move_down: Move vertically (20-{self.max_altitude} cm)
- move_forward/move_back: Move horizontally forward/backward (20-{self.max_distance} cm)
- move_left/move_right: Move horizontally left/right (20-{self.max_distance} cm)
- rotate_clockwise/rotate_counter_clockwise: Rotate (15-180 degrees)
- flip_forward/flip_back/flip_left/flip_right: Perform flips
- start_video/stop_video: Control video streaming
- get_status: Get sensor data and drone status

IMPORTANT: You can now parse MULTIPLE commands from a single input.

Safety rules:
- Maximum movement distance: {self.max_distance} cm
- Maximum altitude movement: {self.max_altitude} cm
- Minimum movement: 20 cm, minimum rotation: 15 degrees
- Flips only while flying
- Emergency always takes priority

Parse the user's command and respond with a JSON array of command objects:
[
    {{
        "action": "command_name",
        "parameters": {{"distance": 50, "degrees": 30}},
        "sequence_order": 1,
        "safety_check": true/false,
        "explanation": "What this step will do"
    }},
    {{
        "action": "second_command",
        "parameters": {{"degrees": 45}},
        "sequence_order": 2,
        "safety_check": true/false,
        "explanation": "What the second step will do"
    }}
]

Rules for multi-commands:
1. Break compound commands into individual actions
2. Assign sequence_order starting from 1
3. Each action should be safe and logical
4. If any action is unsafe, mark safety_check as false
5. Maximum 5 commands per sequence
6. Always prioritize safety over complexity

Examples:
- "move forward 50cm and turn right 30 degrees" = [move_forward, rotate_clockwise]
- "take off, move up 40cm, then turn left" = [takeoff, move_up, rotate_counter_clockwise]
- "show status and move back 30cm" = [get_status, move_back]
"""

        user_prompt = f"Parse this drone command sequence: '{user_input}'"
        
        print(f"🤖 Processing multi-command: {user_input}")
        response = self.query_ollama(user_prompt, system_prompt)
        
        try:
            # Try to extract JSON array from response
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_str = response[start:end]
                commands = json.loads(json_str)
                
                # Ensure it's a list
                if not isinstance(commands, list):
                    commands = [commands]
                
                # Sort by sequence_order
                commands.sort(key=lambda x: x.get('sequence_order', 0))
                
                return commands
            else:
                # Fallback to single command format
                single_cmd = self.parse_command(user_input)
                return [single_cmd] if single_cmd.get('action') != 'none' else []
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            # Fallback to single command
            single_cmd = self.parse_command(user_input)
            return [single_cmd] if single_cmd.get('action') != 'none' else []

    def execute_command_sequence(self, commands: List[Dict[str, Any]]) -> bool:
        """Execute a sequence of commands with safety checks."""
        if not self.tello or not self.tello.connected:
            print("❌ Drone not connected")
            return False
        
        if len(commands) > 5:
            print("⚠️ Too many commands in sequence (max 5). Executing first 5.")
            commands = commands[:5]
        
        print(f"🎯 Executing sequence of {len(commands)} commands:")
        
        # Pre-validate all commands for safety
        validation_results = []
        for i, cmd in enumerate(commands):
            validation = self.validate_command_safety(cmd.get('action', ''), cmd.get('parameters', {}))
            validation_results.append(validation)
            
            if not validation['safe']:
                print(f"❌ Command {i+1} failed safety check:")
                for issue in validation['issues']:
                    print(f"   - {issue}")
                
                if self.safety_mode:
                    print("🛡️ Sequence blocked by safety mode")
                    return False
        
        # Execute commands in sequence
        success_count = 0
        for i, cmd in enumerate(commands):
            action = cmd.get('action', '')
            explanation = cmd.get('explanation', '')
            
            print(f"\n📋 Step {i+1}: {action}")
            print(f"   📝 {explanation}")
            
            # Add delay between commands for safety
            if i > 0:
                print("   ⏳ Waiting 2 seconds before next command...")
                time.sleep(2)
            
            success = self.execute_command(cmd)
            if success:
                success_count += 1
                print(f"   ✅ Step {i+1} completed")
            else:
                print(f"   ❌ Step {i+1} failed")
                
                # Ask user if they want to continue
                if i < len(commands) - 1:
                    try:
                        continue_choice = input(f"   ⚠️ Continue with remaining {len(commands) - i - 1} commands? (y/n): ").lower()
                        if continue_choice != 'y':
                            print("   🛑 Sequence aborted by user")
                            break
                    except KeyboardInterrupt:
                        print("\n   🛑 Sequence interrupted")
                        break
        
        print(f"\n📊 Sequence completed: {success_count}/{len(commands)} commands successful")
        return success_count == len(commands)

    def process_natural_language_command(self, user_input: str) -> bool:
        """Process a natural language command that may contain multiple actions."""
        # Parse the command(s) using Ollama
        commands = self.parse_multi_commands(user_input)
        
        if not commands:
            print("❌ No valid commands found")
            return False
        
        # Log the interaction
        self.conversation_history.append({
            "user": user_input,
            "assistant": f"Parsed {len(commands)} commands",
            "timestamp": datetime.now().isoformat(),
            "commands": [cmd.get('action', '') for cmd in commands]
        })
        
        # Execute the command sequence
        if len(commands) == 1:
            print("🎯 Executing single command")
            return self.execute_command(commands[0])
        else:
            print(f"🎯 Executing command sequence ({len(commands)} commands)")
            return self.execute_command_sequence(commands)
    
    def configure_safety(self):
        """Interactive safety configuration."""
        print("\n🛡️ Safety Configuration")
        print("="*30)
        print(f"Current settings:")
        print(f"  • Safety mode: {'ON' if self.safety_mode else 'OFF'}")
        print(f"  • Max movement distance: {self.max_distance} cm")
        print(f"  • Max altitude change: {self.max_altitude} cm")
        print("="*30)
        
        try:
            # Safety mode toggle
            safety_choice = input("Enable safety mode? (y/n): ").lower()
            self.safety_mode = safety_choice == 'y'
            
            # Distance limits
            try:
                max_dist = input(f"Max movement distance [current: {self.max_distance}cm]: ").strip()
                if max_dist and max_dist.isdigit():
                    new_dist = int(max_dist)
                    if 20 <= new_dist <= 500:
                        self.max_distance = new_dist
                    else:
                        print("⚠️ Distance must be between 20-500cm")
            except ValueError:
                print("⚠️ Invalid distance value")
            
            # Altitude limits
            try:
                max_alt = input(f"Max altitude change [current: {self.max_altitude}cm]: ").strip()
                if max_alt and max_alt.isdigit():
                    new_alt = int(max_alt)
                    if 20 <= new_alt <= 500:
                        self.max_altitude = new_alt
                    else:
                        print("⚠️ Altitude must be between 20-500cm")
            except ValueError:
                print("⚠️ Invalid altitude value")
            
            print("\n✅ Safety configuration updated!")
            print(f"  • Safety mode: {'ON' if self.safety_mode else 'OFF'}")
            print(f"  • Max movement distance: {self.max_distance} cm")
            print(f"  • Max altitude change: {self.max_altitude} cm")
            
        except KeyboardInterrupt:
            print("\n⚠️ Configuration cancelled")

    def emergency_landing_sequence(self):
        """Safe emergency landing with status check."""
        print("🚨 EMERGENCY LANDING SEQUENCE INITIATED")
        
        try:
            # Stop keepalive immediately
            self.stop_keepalive()
            
            # Check current status
            if self.tello and self.tello.connected:
                print("📡 Checking drone status...")
                
                # Try to land safely first
                print("🛬 Attempting safe landing...")
                if self.tello.land():
                    print("✅ Safe landing completed")
                    return True
                else:
                    # If safe landing fails, try emergency stop
                    print("⚠️ Safe landing failed, executing emergency stop...")
                    self.tello.emergency_stop()
                    print("🛑 Emergency stop executed")
                    return True
            else:
                print("❌ Drone not connected")
                return False
                
        except Exception as e:
            print(f"❌ Emergency landing failed: {e}")
            try:
                # Last resort - emergency stop
                if self.tello:
                    self.tello.emergency_stop()
                    print("🛑 Emergency stop executed as last resort")
            except:
                print("❌ All emergency procedures failed")
            return False

    def start_voice_mode(self):
        """Start interactive voice/text mode for natural language control."""
        print("🎤 AI Drone Control Mode Started!")
        print("="*50)
        print("You can now control the drone with natural language!")
        print("Examples:")
        print("  - 'Take off and hover'")
        print("  - 'Move forward 50 centimeters and turn right 30 degrees'")
        print("  - 'Do a front flip then move back'")
        print("  - 'Show me the sensor data'")
        print("  - 'Land safely'")
        print("\nSpecial commands:")
        print("  - 'help' - Show detailed help")
        print("  - 'safety' - Configure safety settings")
        print("  - 'emergency' - Emergency landing sequence")
        print("  - 'quit/exit' - Stop AI control")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n🎯 Command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop', 'bye']:
                    print("👋 Stopping AI control mode...")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'safety':
                    self.configure_safety()
                    continue
                
                if user_input.lower() in ['emergency', 'emergency landing']:
                    self.emergency_landing_sequence()
                    continue
                
                if user_input:
                    success = self.process_natural_language_command(user_input)
                    if success:
                        print("✅ Command sequence completed successfully")
                    else:
                        print("❌ Command sequence failed or was blocked")
                
            except KeyboardInterrupt:
                print("\n🚨 Keyboard interrupt detected!")
                print("Initiating emergency landing sequence...")
                self.emergency_landing_sequence()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Type 'emergency' for emergency landing or 'quit' to exit")
    
    def show_help(self):
        """Show help information."""
        print("\n📚 AI Drone Control Help")
        print("="*40)
        print("Single command examples:")
        print("  • 'Take off' - Launch the drone")
        print("  • 'Move up 50 cm' - Ascend 50 centimeters")
        print("  • 'Turn right 30 degrees' - Rotate clockwise")
        print("  • 'Do a backflip' - Perform backward flip")
        print("\nMulti-command examples:")
        print("  • 'Move forward 40cm and turn left 45 degrees'")
        print("  • 'Take off, move up 60cm, then rotate right'")
        print("  • 'Show status and move back 30cm'")
        print("  • 'Move left 50cm, turn around, then move forward'")
        print("\nSafety features:")
        print(f"  • Max movement: {self.max_distance} cm")
        print(f"  • Max altitude change: {self.max_altitude} cm")
        print(f"  • Min movement: 20 cm, min rotation: 15°")
        print(f"  • Safety mode: {'ON' if self.safety_mode else 'OFF'}")
        print(f"  • Auto-landing prevention: Keepalive every {self.keepalive_interval}s")
        print(f"  • Max commands per sequence: 5")
        print(f"  • Inter-command delay: 2 seconds")
        print("="*40)


def main():
    """Main function to start the AI drone controller."""
    print("🤖 OLLAMA AI DRONE CONTROLLER")
    print("="*50)
    
    # Initialize controller
    ai_controller = OllamaController()
    
    # Check Ollama connection
    print("🔍 Checking Ollama connection...")
    if not ai_controller.check_ollama_connection():
        print("❌ Ollama server not accessible!")
        print("💡 Make sure Ollama is running: 'ollama serve'")
        return
    
    print("✅ Ollama connected!")
    
    # Show available models
    models = ai_controller.get_available_models()
    print(f"📚 Available models: {', '.join(models)}")
    
    if ai_controller.model not in [m.split(':')[0] for m in models]:
        print(f"⚠️ Model '{ai_controller.model}' not found")
        if models:
            print(f"💡 Try: ollama pull {ai_controller.model}")
        return
    
    # Connect to drone
    print("\n🚁 Connecting to Tello drone...")
    if not ai_controller.connect_drone():
        print("❌ Failed to connect to drone!")
        return
    
    print("✅ Drone connected!")
    
    try:
        # Start AI control mode
        ai_controller.start_voice_mode()
    
    finally:
        # Cleanup
        print("\n🔌 Disconnecting from drone...")
        ai_controller.disconnect_drone()
        print("✅ Disconnected successfully!")


if __name__ == "__main__":
    main() 