"""
Neura-Orbit-Agent: Advanced AI Agent for Screen Monitoring and System Automation

A powerful, cross-platform AI agent that can monitor your screen in real-time,
understand what's happening, and perform automated tasks using various LLM models.
"""

__version__ = "0.1.0"
__author__ = "Bayram Eker"
__email__ = "bayrameker@example.com"
__description__ = "Advanced AI Agent for Screen Monitoring and System Automation"

# Core imports for easy access
from .core.agent_brain import NeuraOrbitAgent
from .core.screen_capture import ScreenCapture
from .core.llm_manager import LLMManager
from .core.system_controller import SystemController

# Configuration
from .utils.config import Config
from .utils.logger import get_logger

__all__ = [
    "NeuraOrbitAgent",
    "ScreenCapture", 
    "LLMManager",
    "SystemController",
    "Config",
    "get_logger",
    "__version__",
]
