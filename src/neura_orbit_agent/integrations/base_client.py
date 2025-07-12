"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.name = self.__class__.__name__.replace("Client", "").lower()
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def analyze_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze image with text prompt.
        
        Args:
            image: PIL Image or base64 string
            prompt: Analysis prompt
            model: Model name to use
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List of model information dictionaries
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the client is healthy and can make requests.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_model_capabilities(self, model: str) -> List[str]:
        """
        Get capabilities of a specific model.
        
        Args:
            model: Model name
            
        Returns:
            List of capabilities
        """
        # Default implementation - should be overridden by subclasses
        return ["text"]
    
    def estimate_cost(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None
    ) -> float:
        """
        Estimate cost for a request.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            
        Returns:
            Estimated cost in USD
        """
        # Default implementation returns 0 (free models)
        return 0.0
    
    def get_context_length(self, model: str) -> int:
        """
        Get context length for a model.
        
        Args:
            model: Model name
            
        Returns:
            Context length in tokens
        """
        # Default implementation
        return 4096
