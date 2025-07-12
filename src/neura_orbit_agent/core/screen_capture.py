"""Screen capture and image processing functionality."""

import asyncio
import base64
import io
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mss
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..utils.config import Config
from ..utils.exceptions import ScreenCaptureError
from ..utils.logger import get_screen_logger

logger = get_screen_logger()


class ScreenCapture:
    """Cross-platform screen capture and image processing."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize screen capture.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        self.screen_config = self.config.screen
        
        # Initialize MSS for screen capture
        self.sct = mss.mss()
        
        # Get monitor information
        self.monitors = self.sct.monitors
        self.primary_monitor = self.monitors[1] if len(self.monitors) > 1 else self.monitors[0]

        # Screen size for compatibility
        self.screen_size = (self.primary_monitor["width"], self.primary_monitor["height"])
        
        # Frame buffer for continuous capture
        self.frame_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = self.screen_config.buffer_size
        
        # Capture state
        self.is_capturing = False
        self.capture_task: Optional[asyncio.Task] = None
        
        logger.info(f"Screen capture initialized with {len(self.monitors)-1} monitors")
    
    def get_monitor_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available monitors.
        
        Returns:
            List of monitor information dictionaries
        """
        monitor_info = []
        for i, monitor in enumerate(self.monitors[1:], 1):  # Skip the "All in One" monitor
            info = {
                "id": i,
                "left": monitor["left"],
                "top": monitor["top"], 
                "width": monitor["width"],
                "height": monitor["height"],
                "is_primary": i == 1,
            }
            monitor_info.append(info)
        
        return monitor_info
    
    def capture_screenshot(
        self,
        monitor: int = 0,
        region: Optional[Dict[str, int]] = None,
        format: str = "PNG",
        quality: int = 85,
    ) -> Image.Image:
        """
        Capture a screenshot.
        
        Args:
            monitor: Monitor index (0 for primary, -1 for all)
            region: Optional region to capture {"left": x, "top": y, "width": w, "height": h}
            format: Image format (PNG, JPEG, WEBP)
            quality: Compression quality (1-100)
            
        Returns:
            PIL Image object
            
        Raises:
            ScreenCaptureError: If capture fails
        """
        try:
            # Determine monitor to capture
            if monitor == -1:
                # Capture all monitors
                monitor_dict = self.monitors[0]  # "All in One" monitor
            elif monitor == 0:
                # Primary monitor
                monitor_dict = self.primary_monitor
            else:
                # Specific monitor
                if monitor >= len(self.monitors):
                    raise ScreenCaptureError(f"Monitor {monitor} not found")
                monitor_dict = self.monitors[monitor]
            
            # Apply region if specified
            if region:
                capture_region = {
                    "left": monitor_dict["left"] + region.get("left", 0),
                    "top": monitor_dict["top"] + region.get("top", 0),
                    "width": region.get("width", monitor_dict["width"]),
                    "height": region.get("height", monitor_dict["height"]),
                }
            else:
                capture_region = monitor_dict
            
            # Capture screenshot
            screenshot = self.sct.grab(capture_region)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Apply resolution scaling if needed
            if self.screen_config.resolution != "auto":
                target_resolution = self._parse_resolution(self.screen_config.resolution)
                if target_resolution:
                    img = img.resize(target_resolution, Image.Resampling.LANCZOS)
            
            # Apply privacy filters
            if self.screen_config.blur_sensitive_areas:
                img = self._apply_privacy_filters(img)
            
            logger.debug(f"Screenshot captured: {img.size}")
            return img
            
        except Exception as e:
            raise ScreenCaptureError(f"Failed to capture screenshot: {e}")
    
    def capture_to_base64(
        self,
        monitor: int = 0,
        region: Optional[Dict[str, int]] = None,
        format: str = "PNG",
        quality: int = 85,
    ) -> str:
        """
        Capture screenshot and return as base64 string.
        
        Args:
            monitor: Monitor index
            region: Optional region to capture
            format: Image format
            quality: Compression quality
            
        Returns:
            Base64 encoded image string
        """
        img = self.capture_screenshot(monitor, region, format, quality)
        return self.image_to_base64(img, format, quality)
    
    def capture_to_file(
        self,
        filepath: Union[str, Path],
        monitor: int = 0,
        region: Optional[Dict[str, int]] = None,
        format: Optional[str] = None,
        quality: int = 85,
    ) -> Path:
        """
        Capture screenshot and save to file.
        
        Args:
            filepath: Path to save the image
            monitor: Monitor index
            region: Optional region to capture
            format: Image format (auto-detected from extension if None)
            quality: Compression quality
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format from extension
        if format is None:
            format = filepath.suffix.upper().lstrip(".")
            if format not in ["PNG", "JPEG", "JPG", "WEBP"]:
                format = "PNG"
        
        img = self.capture_screenshot(monitor, region, format, quality)
        
        # Save with appropriate options
        save_kwargs = {}
        if format in ["JPEG", "JPG"]:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif format == "PNG":
            save_kwargs["optimize"] = True
        elif format == "WEBP":
            save_kwargs["quality"] = quality
            save_kwargs["method"] = 6
        
        img.save(filepath, format=format, **save_kwargs)
        logger.info(f"Screenshot saved to {filepath}")
        return filepath
    
    async def start_continuous_capture(
        self,
        interval: Optional[float] = None,
        monitor: int = 0,
        callback: Optional[callable] = None,
    ) -> None:
        """
        Start continuous screen capture.
        
        Args:
            interval: Capture interval in seconds
            monitor: Monitor to capture
            callback: Optional callback function for each frame
        """
        if self.is_capturing:
            logger.warning("Continuous capture already running")
            return
        
        interval = interval or self.screen_config.capture_interval
        self.is_capturing = True
        
        logger.info(f"Starting continuous capture (interval: {interval}s)")
        
        try:
            while self.is_capturing:
                start_time = time.time()
                
                # Capture frame
                img = self.capture_screenshot(monitor)
                
                # Create frame data
                frame_data = {
                    "timestamp": start_time,
                    "image": img,
                    "monitor": monitor,
                    "size": img.size,
                }
                
                # Add to buffer
                self._add_to_buffer(frame_data)
                
                # Call callback if provided
                if callback:
                    try:
                        await callback(frame_data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Continuous capture error: {e}")
        finally:
            self.is_capturing = False
            logger.info("Continuous capture stopped")
    
    def stop_continuous_capture(self) -> None:
        """Stop continuous screen capture."""
        if self.is_capturing:
            self.is_capturing = False
            logger.info("Stopping continuous capture")
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest captured frame.
        
        Returns:
            Latest frame data or None if no frames available
        """
        return self.frame_buffer[-1] if self.frame_buffer else None
    
    def get_frame_buffer(self) -> List[Dict[str, Any]]:
        """
        Get the current frame buffer.
        
        Returns:
            List of frame data
        """
        return self.frame_buffer.copy()
    
    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        logger.debug("Frame buffer cleared")
    
    @staticmethod
    def image_to_base64(img: Image.Image, format: str = "PNG", quality: int = 85) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            img: PIL Image object
            format: Image format
            quality: Compression quality
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        
        save_kwargs = {}
        if format in ["JPEG", "JPG"]:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif format == "PNG":
            save_kwargs["optimize"] = True
        elif format == "WEBP":
            save_kwargs["quality"] = quality
            save_kwargs["method"] = 6
        
        img.save(buffer, format=format, **save_kwargs)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Image.Image:
        """
        Convert base64 string to PIL Image.
        
        Args:
            base64_str: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))
    
    def _add_to_buffer(self, frame_data: Dict[str, Any]) -> None:
        """Add frame to buffer with size management."""
        self.frame_buffer.append(frame_data)
        
        # Remove old frames if buffer is full
        while len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def _parse_resolution(self, resolution_str: str) -> Optional[Tuple[int, int]]:
        """Parse resolution string like '1920x1080'."""
        try:
            width, height = resolution_str.split("x")
            return (int(width), int(height))
        except (ValueError, AttributeError):
            return None
    
    def _apply_privacy_filters(self, img: Image.Image) -> Image.Image:
        """Apply privacy filters to blur sensitive areas."""
        # This is a placeholder - in a real implementation, you would
        # use computer vision to detect sensitive areas like passwords,
        # personal information, etc.
        
        # For now, just blur specified regions
        for region in self.screen_config.exclude_regions:
            # Create a mask for the region
            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([
                region["x"],
                region["y"],
                region["x"] + region["width"],
                region["y"] + region["height"]
            ], fill=255)
            
            # Apply blur to the region
            blurred = img.filter(ImageFilter.GaussianBlur(radius=10))
            img = Image.composite(blurred, img, mask)
        
        return img
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "sct"):
            self.sct.close()
