"""Cross-platform system control and automation."""

import asyncio
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
import pyautogui

from ..utils.config import Config
from ..utils.exceptions import AutomationError, SecurityError
from ..utils.logger import get_automation_logger

logger = get_automation_logger()

# Disable pyautogui failsafe for automation
pyautogui.FAILSAFE = False


class SystemController:
    """Cross-platform system control and automation."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize system controller.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        self.automation_config = self.config.automation
        self.security_config = self.config.security
        
        # System information
        self.platform = platform.system().lower()
        self.is_windows = self.platform == "windows"
        self.is_macos = self.platform == "darwin"
        self.is_linux = self.platform == "linux"
        
        # Screen size for automation
        self.screen_size = pyautogui.size()
        
        logger.info(f"System controller initialized for {self.platform}")
    
    # Mouse and Keyboard Control
    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
        confirm: bool = True
    ) -> bool:
        """
        Click at specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button (left, right, middle)
            clicks: Number of clicks
            interval: Interval between clicks
            confirm: Require user confirmation
            
        Returns:
            True if successful
        """
        if confirm and self.security_config.require_confirmation:
            if not await self._request_confirmation(f"Click at ({x}, {y})"):
                return False
        
        try:
            pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)
            logger.info(f"Clicked at ({x}, {y}) with {button} button")
            return True
        except Exception as e:
            logger.error(f"Failed to click: {e}")
            return False
    
    async def type_text(
        self,
        text: str,
        interval: float = 0.01,
        confirm: bool = True
    ) -> bool:
        """
        Type text at current cursor position.
        
        Args:
            text: Text to type
            interval: Interval between keystrokes
            confirm: Require user confirmation
            
        Returns:
            True if successful
        """
        if confirm and self.security_config.require_confirmation:
            if not await self._request_confirmation(f"Type text: '{text[:50]}...'"):
                return False
        
        try:
            pyautogui.typewrite(text, interval=interval)
            logger.info(f"Typed text: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False
    
    async def press_key(
        self,
        key: Union[str, List[str]],
        confirm: bool = True
    ) -> bool:
        """
        Press keyboard key(s).
        
        Args:
            key: Key name or list of keys for combination
            confirm: Require user confirmation
            
        Returns:
            True if successful
        """
        if confirm and self.security_config.require_confirmation:
            key_str = "+".join(key) if isinstance(key, list) else key
            if not await self._request_confirmation(f"Press key: {key_str}"):
                return False
        
        try:
            if isinstance(key, list):
                pyautogui.hotkey(*key)
            else:
                pyautogui.press(key)
            
            key_str = "+".join(key) if isinstance(key, list) else key
            logger.info(f"Pressed key: {key_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            return False
    
    async def scroll(
        self,
        clicks: int,
        x: Optional[int] = None,
        y: Optional[int] = None
    ) -> bool:
        """
        Scroll at specified position.
        
        Args:
            clicks: Number of scroll clicks (positive = up, negative = down)
            x: X coordinate (current position if None)
            y: Y coordinate (current position if None)
            
        Returns:
            True if successful
        """
        try:
            pyautogui.scroll(clicks, x=x, y=y)
            logger.info(f"Scrolled {clicks} clicks at ({x}, {y})")
            return True
        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            return False
    
    # Application Management
    async def open_application(
        self,
        app_name: str,
        confirm: bool = True
    ) -> bool:
        """
        Open an application.
        
        Args:
            app_name: Application name or path
            confirm: Require user confirmation
            
        Returns:
            True if successful
        """
        if confirm and self.security_config.require_confirmation:
            if not await self._request_confirmation(f"Open application: {app_name}"):
                return False
        
        try:
            if self.is_windows:
                subprocess.Popen(["start", app_name], shell=True)
            elif self.is_macos:
                subprocess.Popen(["open", "-a", app_name])
            else:  # Linux
                subprocess.Popen([app_name])
            
            logger.info(f"Opened application: {app_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to open application {app_name}: {e}")
            return False
    
    async def close_application(
        self,
        app_name: str,
        force: bool = False,
        confirm: bool = True
    ) -> bool:
        """
        Close an application.
        
        Args:
            app_name: Application name
            force: Force close if True
            confirm: Require user confirmation
            
        Returns:
            True if successful
        """
        if confirm and self.security_config.require_confirmation:
            action = "Force close" if force else "Close"
            if not await self._request_confirmation(f"{action} application: {app_name}"):
                return False
        
        try:
            # Find processes by name
            processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                if app_name.lower() in proc.info['name'].lower():
                    processes.append(proc)
            
            if not processes:
                logger.warning(f"No processes found for {app_name}")
                return False
            
            for proc in processes:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                
                logger.info(f"{'Killed' if force else 'Terminated'} process: {proc.info['name']} (PID: {proc.info['pid']})")
            
            return True
        except Exception as e:
            logger.error(f"Failed to close application {app_name}: {e}")
            return False
    
    async def get_running_applications(self) -> List[Dict[str, Any]]:
        """
        Get list of running applications.
        
        Returns:
            List of application information
        """
        try:
            apps = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                try:
                    app_info = {
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'cpu_percent': proc.info['cpu_percent']
                    }
                    apps.append(app_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return apps
        except Exception as e:
            logger.error(f"Failed to get running applications: {e}")
            return []
    
    # Window Management
    async def get_active_window(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the active window.
        
        Returns:
            Window information or None
        """
        try:
            if self.is_windows:
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                window_text = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                
                return {
                    'title': window_text,
                    'x': rect[0],
                    'y': rect[1],
                    'width': rect[2] - rect[0],
                    'height': rect[3] - rect[1]
                }
            elif self.is_macos:
                # macOS implementation would require additional dependencies
                # For now, return basic info
                return {'title': 'Active Window', 'x': 0, 'y': 0, 'width': 800, 'height': 600}
            else:
                # Linux implementation would require X11 libraries
                return {'title': 'Active Window', 'x': 0, 'y': 0, 'width': 800, 'height': 600}
                
        except Exception as e:
            logger.error(f"Failed to get active window: {e}")
            return None
    
    # File Operations
    async def open_file(
        self,
        file_path: Union[str, Path],
        confirm: bool = True
    ) -> bool:
        """
        Open a file with default application.
        
        Args:
            file_path: Path to file
            confirm: Require user confirmation
            
        Returns:
            True if successful
        """
        file_path = Path(file_path)
        
        if confirm and self.security_config.require_confirmation:
            if not await self._request_confirmation(f"Open file: {file_path}"):
                return False
        
        try:
            if self.is_windows:
                subprocess.Popen(["start", str(file_path)], shell=True)
            elif self.is_macos:
                subprocess.Popen(["open", str(file_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(file_path)])
            
            logger.info(f"Opened file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return False
    
    # System Information
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            System information dictionary
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'platform': self.platform,
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': cpu_percent,
                'memory_total_gb': memory.total / 1024**3,
                'memory_used_gb': memory.used / 1024**3,
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / 1024**3,
                'disk_used_gb': disk.used / 1024**3,
                'disk_percent': (disk.used / disk.total) * 100,
                'screen_size': self.screen_size
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
    
    # Security and Confirmation
    async def _request_confirmation(self, action: str) -> bool:
        """
        Request user confirmation for an action.
        
        Args:
            action: Description of action
            
        Returns:
            True if confirmed
        """
        # In a real implementation, this would show a GUI dialog or CLI prompt
        # For now, we'll assume confirmation is granted
        logger.info(f"Action requires confirmation: {action}")
        
        # Simulate user confirmation (in real implementation, this would be interactive)
        await asyncio.sleep(0.1)  # Simulate user thinking time
        return True
    
    def _check_permission(self, action: str) -> bool:
        """
        Check if action is permitted.
        
        Args:
            action: Action to check
            
        Returns:
            True if permitted
        """
        permissions = self.security_config.permissions
        
        if action in permissions.get("safe_actions", []):
            return True
        elif action in permissions.get("moderate_actions", []):
            return True  # Could add additional checks here
        elif action in permissions.get("restricted_actions", []):
            return False  # Restricted actions not allowed
        
        # Default to safe for unknown actions
        return True
