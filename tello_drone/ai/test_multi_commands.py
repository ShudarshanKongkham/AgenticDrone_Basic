#!/usr/bin/env python3
"""
Test script for multi-command parsing functionality.
Tests the AI controller without requiring drone connection.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ollama_controller import OllamaController


def test_multi_command_parsing():
    """Test multi-command parsing functionality."""
    print("🧪 Testing Multi-Command Parsing")
    print("="*50)
    
    # Initialize controller (without drone connection)
    ai_controller = OllamaController()
    
    # Test commands
    test_commands = [
        "move forward 50cm and turn right 30 degrees",
        "Move up in a zig zag pattern, and the move down to the intial postion and move forward 50cm",
        "Move up in a zig zag pattern by moving left and right, and the move down to the intial postion and move forward 50cm",
        "take off, move up 40cm, then rotate left",
        "fly in a circle for 10 seconds",
        "move forward lika a snake",
        "show status and move back 25cm",
        "move left 60cm, turn around, then move forward 30cm",
        "do a front flip and land",
        "emergency stop",  # Single command
        "move up 20cm",    # Single command
    ]
    
    print("🔍 Testing command parsing (without execution):")
    print()
    
    for i, command in enumerate(test_commands, 1):
        print(f"Test {i}: '{command}'")
        print("-" * 40)
        
        try:
            # Parse the multi-command
            commands = ai_controller.parse_multi_commands(command)
            
            if commands:
                print(f"✅ Parsed {len(commands)} command(s):")
                for j, cmd in enumerate(commands, 1):
                    action = cmd.get('action', 'unknown')
                    params = cmd.get('parameters', {})
                    explanation = cmd.get('explanation', 'No explanation')
                    safety = cmd.get('safety_check', False)
                    
                    print(f"  {j}. Action: {action}")
                    print(f"     Parameters: {params}")
                    print(f"     Safe: {'✅' if safety else '❌'}")
                    print(f"     Explanation: {explanation}")
                    
                    # Test safety validation
                    validation = ai_controller.validate_command_safety(action, params)
                    if not validation['safe']:
                        print(f"     ⚠️ Safety issues: {', '.join(validation['issues'])}")
                    print()
            else:
                print("❌ No commands parsed")
                
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
        
        print("=" * 50)
        print()


def test_safety_validation():
    """Test safety validation functionality."""
    print("\n🛡️ Testing Safety Validation")
    print("="*40)
    
    ai_controller = OllamaController()
    
    # Test cases: (action, parameters, expected_safe)
    test_cases = [
        ("move_forward", {"distance": 50}, True),
        ("move_forward", {"distance": 150}, False),  # Too far
        ("move_forward", {"distance": 10}, False),   # Too close
        ("move_up", {"distance": 30}, True),
        ("move_up", {"distance": 250}, False),       # Too high
        ("rotate_clockwise", {"degrees": 45}, True),
        ("rotate_clockwise", {"degrees": 200}, False), # Too much rotation
        ("rotate_clockwise", {"degrees": 5}, False),   # Too little rotation
        ("flip_forward", {}, True),  # Would need flying check in real scenario
        ("emergency", {}, True),     # Always safe
    ]
    
    for action, params, expected_safe in test_cases:
        validation = ai_controller.validate_command_safety(action, params)
        actual_safe = validation['safe']
        
        status = "✅" if actual_safe == expected_safe else "❌"
        print(f"{status} {action} {params}: {'Safe' if actual_safe else 'Unsafe'}")
        
        if not actual_safe:
            print(f"   Issues: {', '.join(validation['issues'])}")


def main():
    """Run all tests."""
    print("🤖 OLLAMA CONTROLLER MULTI-COMMAND TESTS")
    print("="*60)
    print("Note: These tests parse commands without drone execution")
    print("="*60)
    
    # Check if Ollama is available
    ai_controller = OllamaController()
    if not ai_controller.check_ollama_connection():
        print("⚠️ Ollama not available - using mock responses")
        print("💡 Start Ollama with: 'ollama serve'")
        print("💡 Install model with: 'ollama pull granite3.3'")
        return
    
    print("✅ Ollama connected!")
    print()
    
    try:
        test_multi_command_parsing()
        test_safety_validation()
        
        print("🎉 All tests completed!")
        print("\n💡 To test with real drone:")
        print("   1. Connect Tello drone")
        print("   2. Run: python ollama_controller.py")
        print("   3. Try: 'move forward 40cm and turn left 30 degrees'")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main() 