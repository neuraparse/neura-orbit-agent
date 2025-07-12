"""OpenAI client for GPT models."""

import base64
import io
from typing import Any, Dict, List, Optional, Union

import openai
from PIL import Image

from .base_client import BaseLLMClient
from ..utils.exceptions import LLMError, ModelNotFoundError
from ..utils.logger import get_llm_logger

logger = get_llm_logger()


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI client.
        
        Args:
            config: OpenAI configuration
        """
        super().__init__(config)
        
        api_key = config.get("api_key")
        if not api_key:
            raise LLMError("OpenAI API key is required")
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            organization=config.get("organization"),
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
        Generate text using OpenAI GPT.
        
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
            model = "gpt-3.5-turbo"
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content or ""
            
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error in OpenAI generation: {e}")
    
    async def analyze_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze image using OpenAI vision model.
        
        Args:
            image: PIL Image or base64 string
            prompt: Analysis prompt
            model: Vision model name
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        if model is None:
            model = "gpt-4-vision-preview"
        
        # Convert image to base64 if needed
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            image_b64 = image
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return response.choices[0].message.content or ""
            
        except openai.APIError as e:
            raise LLMError(f"OpenAI vision API error: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error in OpenAI vision analysis: {e}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available OpenAI models.
        
        Returns:
            List of model information
        """
        try:
            models_response = await self.client.models.list()
            models = []
            
            # Filter for chat models
            chat_models = [
                model for model in models_response.data 
                if "gpt" in model.id.lower()
            ]
            
            for model in chat_models:
                model_config = self.model_configs.get(model.id, {})
                
                model_data = {
                    "name": model.id,
                    "created": model.created,
                    "capabilities": model_config.get("capabilities", ["text", "chat"]),
                    "context_length": model_config.get("context_length", 4096),
                    "cost_per_1k_tokens": model_config.get("cost_per_1k_tokens", 0.002),
                    "provider": "openai"
                }
                models.append(model_data)
            
            logger.info(f"Found {len(models)} OpenAI models")
            return models
            
        except openai.APIError as e:
            raise LLMError(f"Failed to get OpenAI models: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error getting OpenAI models: {e}")
    
    async def health_check(self) -> bool:
        """
        Check OpenAI API health.
        
        Returns:
            True if healthy
        """
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
    
    def get_model_capabilities(self, model: str) -> List[str]:
        """Get capabilities of a specific model."""
        model_config = self.model_configs.get(model, {})
        return model_config.get("capabilities", ["text", "chat"])
    
    def estimate_cost(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None
    ) -> float:
        """
        Estimate cost for OpenAI request.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            
        Returns:
            Estimated cost in USD
        """
        model_config = self.model_configs.get(model, {})
        cost_per_1k = model_config.get("cost_per_1k_tokens", 0.002)
        
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) / 4
        output_tokens = max_tokens or 150
        
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * cost_per_1k
    
    def get_context_length(self, model: str) -> int:
        """Get context length for a model."""
        model_config = self.model_configs.get(model, {})
        return model_config.get("context_length", 4096)
