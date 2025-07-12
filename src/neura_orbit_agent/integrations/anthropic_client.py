"""Anthropic client for Claude models."""

import base64
import io
from typing import Any, Dict, List, Optional, Union

import anthropic
from PIL import Image

from .base_client import BaseLLMClient
from ..utils.exceptions import LLMError, ModelNotFoundError
from ..utils.logger import get_llm_logger

logger = get_llm_logger()


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic client.
        
        Args:
            config: Anthropic configuration
        """
        super().__init__(config)
        
        api_key = config.get("api_key")
        if not api_key:
            raise LLMError("Anthropic API key is required")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=config.get("timeout", 60)
        )
        
        # Model configurations from config
        self.model_configs = {
            model["name"]: model 
            for model in config.get("models", [])
        }
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using Anthropic Claude.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if model is None:
            model = "claude-3-haiku-20240307"
        
        if max_tokens is None:
            max_tokens = 1000
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            return response.content[0].text if response.content else ""
            
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error in Anthropic generation: {e}")
    
    async def analyze_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze image using Anthropic Claude vision.
        
        Args:
            image: PIL Image or base64 string
            prompt: Analysis prompt
            model: Vision model name
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        if model is None:
            model = "claude-3-sonnet-20240229"
        
        # Convert image to base64 if needed
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            image_b64 = image
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return response.content[0].text if response.content else ""
            
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic vision API error: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error in Anthropic vision analysis: {e}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available Anthropic models.
        
        Returns:
            List of model information
        """
        # Anthropic doesn't have a models endpoint, so we return configured models
        models = []
        
        for model_name, model_config in self.model_configs.items():
            model_data = {
                "name": model_name,
                "capabilities": model_config.get("capabilities", ["text", "chat"]),
                "context_length": model_config.get("context_length", 200000),
                "cost_per_1k_tokens": model_config.get("cost_per_1k_tokens", 0.003),
                "provider": "anthropic"
            }
            models.append(model_data)
        
        # If no models configured, return defaults
        if not models:
            default_models = [
                {
                    "name": "claude-3-sonnet-20240229",
                    "capabilities": ["text", "chat", "reasoning", "vision"],
                    "context_length": 200000,
                    "cost_per_1k_tokens": 0.003,
                    "provider": "anthropic"
                },
                {
                    "name": "claude-3-haiku-20240307",
                    "capabilities": ["text", "chat", "fast"],
                    "context_length": 200000,
                    "cost_per_1k_tokens": 0.00025,
                    "provider": "anthropic"
                }
            ]
            models = default_models
        
        logger.info(f"Found {len(models)} Anthropic models")
        return models
    
    async def health_check(self) -> bool:
        """
        Check Anthropic API health.
        
        Returns:
            True if healthy
        """
        try:
            # Simple test request
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception:
            return False
    
    def get_model_capabilities(self, model: str) -> List[str]:
        """Get capabilities of a specific model."""
        model_config = self.model_configs.get(model, {})
        
        # Default capabilities based on model name
        if "sonnet" in model.lower():
            default_caps = ["text", "chat", "reasoning", "vision", "analysis"]
        elif "haiku" in model.lower():
            default_caps = ["text", "chat", "fast"]
        else:
            default_caps = ["text", "chat"]
        
        return model_config.get("capabilities", default_caps)
    
    def estimate_cost(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None
    ) -> float:
        """
        Estimate cost for Anthropic request.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            
        Returns:
            Estimated cost in USD
        """
        model_config = self.model_configs.get(model, {})
        
        # Default costs if not configured
        if "sonnet" in model.lower():
            cost_per_1k = model_config.get("cost_per_1k_tokens", 0.003)
        elif "haiku" in model.lower():
            cost_per_1k = model_config.get("cost_per_1k_tokens", 0.00025)
        else:
            cost_per_1k = model_config.get("cost_per_1k_tokens", 0.003)
        
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) / 4
        output_tokens = max_tokens or 150
        
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * cost_per_1k
    
    def get_context_length(self, model: str) -> int:
        """Get context length for a model."""
        model_config = self.model_configs.get(model, {})
        return model_config.get("context_length", 200000)
