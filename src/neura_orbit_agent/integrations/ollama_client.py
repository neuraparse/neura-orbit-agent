"""Ollama client for local LLM inference."""

import asyncio
import base64
import io
from typing import Any, Dict, List, Optional, Union

import aiohttp
from PIL import Image

from .base_client import BaseLLMClient
from ..utils.exceptions import LLMError, ModelNotFoundError
from ..utils.logger import get_llm_logger

logger = get_llm_logger()


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM server."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama client.
        
        Args:
            config: Ollama configuration
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 60)
        self.models_cache: Optional[List[Dict[str, Any]]] = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model name (defaults to first available)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails
            ModelNotFoundError: If model not found
        """
        if model is None:
            models = await self.get_available_models()
            if not models:
                raise ModelNotFoundError("No models available in Ollama")
            model = models[0]["name"]
        
        session = await self._get_session()
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            data["options"]["num_predict"] = max_tokens
        
        # Add any additional options
        data["options"].update(kwargs)
        
        try:
            async with session.post(f"{self.base_url}/api/generate", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"Ollama API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "error" in result:
                    raise LLMError(f"Ollama error: {result['error']}")
                
                return result.get("response", "")
                
        except aiohttp.ClientError as e:
            raise LLMError(f"Connection error to Ollama: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error in Ollama generation: {e}")
    
    async def analyze_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze image using Ollama vision model.
        
        Args:
            image: PIL Image or base64 string
            prompt: Analysis prompt
            model: Vision model name
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        # Convert image to base64 if needed
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            image_b64 = image
        
        # Find a vision-capable model if none specified
        if model is None:
            models = await self.get_available_models()
            vision_models = [m for m in models if "vision" in m["name"].lower()]
            if not vision_models:
                raise ModelNotFoundError("No vision models available in Ollama")
            model = vision_models[0]["name"]
        
        session = await self._get_session()
        
        data = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": kwargs
        }
        
        try:
            async with session.post(f"{self.base_url}/api/generate", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"Ollama vision API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "error" in result:
                    raise LLMError(f"Ollama vision error: {result['error']}")
                
                return result.get("response", "")
                
        except aiohttp.ClientError as e:
            raise LLMError(f"Connection error to Ollama: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error in Ollama vision analysis: {e}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models from Ollama.
        
        Returns:
            List of model information
        """
        if self.models_cache is not None:
            return self.models_cache
        
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"Failed to get Ollama models: {response.status} - {error_text}")
                
                result = await response.json()
                models = []
                
                for model_info in result.get("models", []):
                    model_data = {
                        "name": model_info["name"],
                        "size": model_info.get("size", 0),
                        "modified": model_info.get("modified_at", ""),
                        "capabilities": self._infer_capabilities(model_info["name"]),
                        "provider": "ollama"
                    }
                    models.append(model_data)
                
                self.models_cache = models
                logger.info(f"Found {len(models)} Ollama models")
                return models
                
        except aiohttp.ClientError as e:
            raise LLMError(f"Connection error to Ollama: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error getting Ollama models: {e}")
    
    async def health_check(self) -> bool:
        """
        Check Ollama server health.
        
        Returns:
            True if healthy
        """
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False
    
    def _infer_capabilities(self, model_name: str) -> List[str]:
        """
        Infer model capabilities from name.
        
        Args:
            model_name: Model name
            
        Returns:
            List of capabilities
        """
        capabilities = ["text"]
        
        name_lower = model_name.lower()
        
        if "vision" in name_lower or "llava" in name_lower:
            capabilities.append("vision")
            capabilities.append("multimodal")
        
        if "code" in name_lower or "coder" in name_lower:
            capabilities.append("coding")
        
        if any(term in name_lower for term in ["instruct", "chat", "assistant"]):
            capabilities.append("chat")
            capabilities.append("reasoning")
        
        return capabilities
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        session = await self._get_session()
        
        data = {"name": model_name}
        
        try:
            async with session.post(f"{self.base_url}/api/pull", json=data) as response:
                if response.status != 200:
                    return False
                
                # Clear models cache to refresh
                self.models_cache = None
                logger.info(f"Successfully pulled model: {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
