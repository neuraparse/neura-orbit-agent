"""Custom exceptions for Neura-Orbit-Agent."""

from typing import Any, Dict, Optional


class NeuraOrbitError(Exception):
    """Base exception for all Neura-Orbit-Agent errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(NeuraOrbitError):
    """Raised when there's a configuration-related error."""
    pass


class LLMError(NeuraOrbitError):
    """Raised when there's an LLM-related error."""
    pass


class ScreenCaptureError(NeuraOrbitError):
    """Raised when there's a screen capture-related error."""
    pass


class AutomationError(NeuraOrbitError):
    """Raised when there's an automation-related error."""
    pass


class SecurityError(NeuraOrbitError):
    """Raised when there's a security-related error."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when a requested model is not available."""
    pass


class PermissionDeniedError(SecurityError):
    """Raised when an action is not permitted."""
    pass


class TaskExecutionError(NeuraOrbitError):
    """Raised when task execution fails."""
    pass


class ValidationError(NeuraOrbitError):
    """Raised when input validation fails."""
    pass
