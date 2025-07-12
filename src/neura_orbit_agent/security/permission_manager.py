"""Permission management for secure operations."""

from typing import Any, Dict, List, Optional

from ..utils.config import Config
from ..utils.exceptions import SecurityError, PermissionDeniedError
from ..utils.logger import get_security_logger

logger = get_security_logger()


class PermissionManager:
    """Manages permissions for system operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize permission manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        self.security_config = self.config.security
        
        logger.info("Permission manager initialized")
    
    def check_permission(self, action: str) -> bool:
        """
        Check if an action is permitted.
        
        Args:
            action: Action to check
            
        Returns:
            True if permitted
            
        Raises:
            PermissionDeniedError: If action is not permitted
        """
        permissions = self.security_config.permissions
        
        if action in permissions.get("safe_actions", []):
            return True
        elif action in permissions.get("moderate_actions", []):
            return True
        elif action in permissions.get("restricted_actions", []):
            raise PermissionDeniedError(f"Action '{action}' is restricted")
        
        # Default to safe for unknown actions
        logger.warning(f"Unknown action '{action}', defaulting to permitted")
        return True
    
    async def request_confirmation(self, action: str, details: str = "") -> bool:
        """
        Request user confirmation for an action.
        
        Args:
            action: Action description
            details: Additional details
            
        Returns:
            True if confirmed
        """
        if not self.security_config.require_confirmation:
            return True
        
        # In a real implementation, this would show a GUI dialog or CLI prompt
        logger.info(f"Action requires confirmation: {action} - {details}")
        
        # For now, assume confirmation is granted
        return True
