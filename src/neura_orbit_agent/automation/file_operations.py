"""File operations automation."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.config import Config
from ..utils.exceptions import AutomationError
from ..utils.logger import get_automation_logger

logger = get_automation_logger()


class FileOperations:
    """File operations automation."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize file operations.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        
        logger.info("File operations initialized")
    
    async def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Read a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content or None if failed
        """
        try:
            file_path = Path(file_path)
            content = file_path.read_text(encoding='utf-8')
            logger.info(f"Read file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    async def write_file(self, file_path: Union[str, Path], content: str) -> bool:
        """
        Write to a file.
        
        Args:
            file_path: Path to file
            content: Content to write
            
        Returns:
            True if successful
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Wrote file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False
