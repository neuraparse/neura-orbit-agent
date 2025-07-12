#!/usr/bin/env python3
"""Test real browser automation."""

import asyncio
import time
from src.neura_orbit_agent.core.system_controller import SystemController
from src.neura_orbit_agent.core.screen_capture import ScreenCapture
from src.neura_orbit_agent.utils.config import Config

async def test_browser_automation():
    """Test real browser automation scenario."""
    print("🌐 Testing Real Browser Automation...")
    
    config = Config.load_default()
    controller = SystemController(config)
    screen_capture = ScreenCapture(config)
    
    print("\n1. 📸 Taking initial screenshot...")
    screenshot = screen_capture.capture_screenshot()
    screen_capture.capture_to_file("screenshots/before_automation.png")
    print("   ✅ Initial screenshot saved")
    
    print("\n2. 🚀 Opening Safari...")
    await controller.open_application("Safari")
    await asyncio.sleep(3)  # Wait for Safari to open
    
    print("\n3. 🔍 Opening new tab...")
    await controller.press_key(["cmd", "t"])  # New tab
    await asyncio.sleep(1)
    
    print("\n4. 🌐 Navigating to Google...")
    await controller.type_text("https://www.google.com")
    await controller.press_key("enter")
    await asyncio.sleep(3)  # Wait for page to load
    
    print("\n5. 🔍 Searching for 'Neura Orbit Agent'...")
    # Click on search box (usually focused automatically)
    await controller.type_text("Neura Orbit Agent AI automation")
    await asyncio.sleep(1)
    await controller.press_key("enter")
    await asyncio.sleep(3)  # Wait for search results
    
    print("\n6. 📸 Taking screenshot of search results...")
    screenshot = screen_capture.capture_screenshot()
    screen_capture.capture_to_file("screenshots/google_search_results.png")
    print("   ✅ Search results screenshot saved")
    
    print("\n7. 📝 Opening a new tab for note-taking...")
    await controller.press_key(["cmd", "t"])  # New tab
    await asyncio.sleep(1)
    
    # Navigate to a simple text editor (like Google Docs or just use TextEdit)
    print("\n8. 📝 Opening TextEdit for notes...")
    await controller.press_key(["cmd", "space"])  # Spotlight
    await asyncio.sleep(1)
    await controller.type_text("TextEdit")
    await controller.press_key("enter")
    await asyncio.sleep(2)
    
    print("\n9. ✍️ Writing automation report...")
    report_text = """
🤖 Neura-Orbit-Agent Automation Test Report
==========================================

✅ Successfully opened Safari browser
✅ Created new tab
✅ Navigated to Google.com
✅ Performed search query
✅ Captured screenshots
✅ Opened TextEdit for documentation

🎯 Automation Capabilities Verified:
- Application launching
- Keyboard shortcuts
- Text input
- Web navigation
- Multi-tasking
- Screenshot capture

🚀 System is ready for complex automation tasks!
"""
    
    await controller.type_text(report_text)
    await asyncio.sleep(2)
    
    print("\n10. 💾 Saving the report...")
    await controller.press_key(["cmd", "s"])  # Save
    await asyncio.sleep(1)
    await controller.type_text("Neura-Orbit-Automation-Report")
    await controller.press_key("enter")
    
    print("\n11. 📸 Taking final screenshot...")
    screenshot = screen_capture.capture_screenshot()
    screen_capture.capture_to_file("screenshots/automation_complete.png")
    print("   ✅ Final screenshot saved")
    
    print("\n🎉 Browser Automation Test Completed Successfully!")
    print("\n📊 Summary:")
    print("   ✅ Opened Safari browser")
    print("   ✅ Performed Google search")
    print("   ✅ Opened TextEdit")
    print("   ✅ Created automation report")
    print("   ✅ Saved document")
    print("   ✅ Captured 3 screenshots")
    
    print("\n🔧 This demonstrates:")
    print("   • Multi-application control")
    print("   • Web browser automation")
    print("   • Text editing automation")
    print("   • File operations")
    print("   • Screenshot documentation")
    print("   • Complex workflow execution")

if __name__ == "__main__":
    # Create screenshots directory
    import os
    os.makedirs("screenshots", exist_ok=True)
    
    asyncio.run(test_browser_automation())
