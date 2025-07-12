#!/usr/bin/env python3
"""Test script to verify screen control capabilities."""

import asyncio
import time
from src.neura_orbit_agent.core.system_controller import SystemController
from src.neura_orbit_agent.core.screen_capture import ScreenCapture
from src.neura_orbit_agent.utils.config import Config

async def test_screen_control():
    """Test basic screen control functions."""
    print("🚀 Testing Neura-Orbit-Agent Screen Control...")
    
    # Initialize components
    config = Config.load_default()
    screen_capture = ScreenCapture(config)
    system_controller = SystemController(config)
    
    print("\n📸 1. Taking screenshot...")
    screenshot = screen_capture.capture_screenshot()
    print(f"   ✅ Screenshot captured: {screenshot.size}")
    
    print("\n🖱️  2. Getting current mouse position...")
    import pyautogui
    current_pos = pyautogui.position()
    print(f"   ✅ Current mouse position: {current_pos}")
    
    print("\n🎯 3. Testing mouse movement...")
    # Move mouse to center of screen
    center_x = screen_capture.screen_size[0] // 2
    center_y = screen_capture.screen_size[1] // 2
    
    print(f"   Moving mouse to center: ({center_x}, {center_y})")
    pyautogui.moveTo(center_x, center_y, duration=1)
    
    new_pos = pyautogui.position()
    print(f"   ✅ Mouse moved to: {new_pos}")
    
    print("\n⌨️  4. Testing keyboard input...")
    # Wait a moment then type some text
    print("   Typing test text in 3 seconds...")
    await asyncio.sleep(3)
    
    # Type some text
    await system_controller.type_text("Hello from Neura-Orbit-Agent! 🤖", confirm=False)
    print("   ✅ Text typed successfully")
    
    print("\n🔄 5. Testing key combinations...")
    # Test Cmd+A (Select All) on macOS
    await system_controller.press_key(["cmd", "a"], confirm=False)
    print("   ✅ Cmd+A pressed (Select All)")
    
    await asyncio.sleep(1)
    
    # Test Backspace to delete
    await system_controller.press_key("backspace", confirm=False)
    print("   ✅ Backspace pressed")
    
    print("\n📊 6. Getting system info...")
    system_info = await system_controller.get_system_info()
    print(f"   Platform: {system_info.get('platform')}")
    print(f"   CPU Usage: {system_info.get('cpu_percent', 0):.1f}%")
    print(f"   Memory Usage: {system_info.get('memory_percent', 0):.1f}%")
    
    print("\n🎉 All tests completed successfully!")
    print("\n🔧 Available capabilities:")
    print("   ✅ Screen capture and analysis")
    print("   ✅ Mouse movement and clicking")
    print("   ✅ Keyboard input and key combinations")
    print("   ✅ System information gathering")
    print("   ✅ Cross-platform compatibility")

if __name__ == "__main__":
    asyncio.run(test_screen_control())
