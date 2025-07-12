"""Tests for screen capture functionality."""

import pytest
from PIL import Image

from neura_orbit_agent.core.screen_capture import ScreenCapture
from neura_orbit_agent.utils.config import Config


class TestScreenCapture:
    """Test cases for ScreenCapture class."""
    
    @pytest.fixture
    def screen_capture(self):
        """Create a ScreenCapture instance for testing."""
        config = Config.load_default()
        return ScreenCapture(config)
    
    def test_initialization(self, screen_capture):
        """Test ScreenCapture initialization."""
        assert screen_capture is not None
        assert screen_capture.sct is not None
        assert len(screen_capture.monitors) > 0
        assert screen_capture.primary_monitor is not None
    
    def test_get_monitor_info(self, screen_capture):
        """Test getting monitor information."""
        monitor_info = screen_capture.get_monitor_info()
        assert isinstance(monitor_info, list)
        assert len(monitor_info) > 0
        
        for monitor in monitor_info:
            assert "id" in monitor
            assert "width" in monitor
            assert "height" in monitor
            assert "left" in monitor
            assert "top" in monitor
            assert "is_primary" in monitor
    
    def test_capture_screenshot(self, screen_capture):
        """Test basic screenshot capture."""
        screenshot = screen_capture.capture_screenshot()
        
        assert isinstance(screenshot, Image.Image)
        assert screenshot.size[0] > 0
        assert screenshot.size[1] > 0
        assert screenshot.mode in ["RGB", "RGBA"]
    
    def test_capture_to_base64(self, screen_capture):
        """Test screenshot capture to base64."""
        base64_str = screen_capture.capture_to_base64()
        
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        
        # Test conversion back to image
        img = ScreenCapture.base64_to_image(base64_str)
        assert isinstance(img, Image.Image)
    
    def test_image_conversion(self):
        """Test image to base64 conversion and back."""
        # Create a test image
        test_img = Image.new("RGB", (100, 100), color="red")
        
        # Convert to base64
        base64_str = ScreenCapture.image_to_base64(test_img)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        
        # Convert back to image
        converted_img = ScreenCapture.base64_to_image(base64_str)
        assert isinstance(converted_img, Image.Image)
        assert converted_img.size == test_img.size
    
    @pytest.mark.asyncio
    async def test_frame_buffer(self, screen_capture):
        """Test frame buffer functionality."""
        # Initially empty
        assert len(screen_capture.frame_buffer) == 0
        
        # Capture a frame
        img = screen_capture.capture_screenshot()
        frame_data = {
            "timestamp": 123456789,
            "image": img,
            "monitor": 0,
            "size": img.size,
        }
        
        screen_capture._add_to_buffer(frame_data)
        assert len(screen_capture.frame_buffer) == 1
        
        # Get latest frame
        latest = screen_capture.get_latest_frame()
        assert latest is not None
        assert latest["timestamp"] == 123456789
        
        # Clear buffer
        screen_capture.clear_buffer()
        assert len(screen_capture.frame_buffer) == 0
