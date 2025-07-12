"""LLM Manager for handling multiple LLM providers and model selection."""

import asyncio
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..integrations.ollama_client import OllamaClient
from ..integrations.openai_client import OpenAIClient
from ..integrations.anthropic_client import AnthropicClient
from ..integrations.base_client import BaseLLMClient
from ..utils.config import Config
from ..utils.exceptions import LLMError, ModelNotFoundError
from ..utils.logger import get_llm_logger

logger = get_llm_logger()


class LLMManager:
    """Manager for multiple LLM providers with intelligent model selection."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LLM Manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        self.llm_config = self.config.llm
        
        # Initialize clients
        self.clients: Dict[str, BaseLLMClient] = {}
        self._initialize_clients()
        
        # Model cache
        self.available_models: Dict[str, List[Dict[str, Any]]] = {}
        self.model_health: Dict[str, bool] = {}
    
    def _initialize_clients(self) -> None:
        """Initialize LLM clients based on configuration."""
        for provider_name, provider_config in self.llm_config.providers.items():
            try:
                if provider_name == "ollama":
                    client = OllamaClient(provider_config.dict())
                elif provider_name == "openai":
                    if provider_config.api_key:
                        client = OpenAIClient(provider_config.dict())
                    else:
                        logger.warning("OpenAI API key not provided, skipping")
                        continue
                elif provider_name == "anthropic":
                    if provider_config.api_key:
                        client = AnthropicClient(provider_config.dict())
                    else:
                        logger.warning("Anthropic API key not provided, skipping")
                        continue
                else:
                    logger.warning(f"Unknown provider: {provider_name}")
                    continue
                
                self.clients[provider_name] = client
                logger.info(f"Initialized {provider_name} client")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} client: {e}")
    
    async def get_available_models(self, refresh: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get available models from all providers.
        
        Args:
            refresh: Force refresh of model cache
            
        Returns:
            Dictionary mapping provider names to model lists
        """
        if not refresh and self.available_models:
            return self.available_models
        
        self.available_models = {}
        
        for provider_name, client in self.clients.items():
            try:
                models = await client.get_available_models()
                self.available_models[provider_name] = models
                self.model_health[provider_name] = True
                logger.debug(f"Retrieved {len(models)} models from {provider_name}")
            except Exception as e:
                logger.error(f"Failed to get models from {provider_name}: {e}")
                self.available_models[provider_name] = []
                self.model_health[provider_name] = False
        
        return self.available_models
    
    async def select_best_model(
        self,
        task_type: str = "general",
        capabilities: Optional[List[str]] = None,
        prefer_local: bool = True,
        max_cost: Optional[float] = None,
    ) -> tuple[str, str]:  # (provider, model)
        """
        Select the best model for a given task.
        
        Args:
            task_type: Type of task (general, vision, coding, reasoning)
            capabilities: Required capabilities
            prefer_local: Prefer local models (Ollama)
            max_cost: Maximum cost per request
            
        Returns:
            Tuple of (provider_name, model_name)
            
        Raises:
            ModelNotFoundError: If no suitable model found
        """
        await self.get_available_models()
        
        capabilities = capabilities or []
        
        # Add task-specific capabilities
        if task_type == "vision":
            capabilities.extend(["vision", "multimodal"])
        elif task_type == "coding":
            capabilities.extend(["coding"])
        elif task_type == "reasoning":
            capabilities.extend(["reasoning"])
        
        suitable_models = []
        
        # Evaluate models from each provider
        for provider_name, models in self.available_models.items():
            if not self.model_health.get(provider_name, False):
                continue
            
            for model in models:
                model_caps = model.get("capabilities", [])
                
                # Check if model has required capabilities
                if capabilities and not all(cap in model_caps for cap in capabilities):
                    continue
                
                # Check cost constraint
                if max_cost is not None:
                    model_cost = model.get("cost_per_1k_tokens", 0)
                    if model_cost > max_cost:
                        continue
                
                # Calculate score
                score = self._calculate_model_score(
                    provider_name, model, task_type, prefer_local
                )
                
                suitable_models.append({
                    "provider": provider_name,
                    "model": model["name"],
                    "score": score,
                    "capabilities": model_caps,
                    "cost": model.get("cost_per_1k_tokens", 0)
                })
        
        if not suitable_models:
            raise ModelNotFoundError(
                f"No suitable model found for task '{task_type}' with capabilities {capabilities}"
            )
        
        # Sort by score (higher is better)
        suitable_models.sort(key=lambda x: x["score"], reverse=True)
        best_model = suitable_models[0]
        
        logger.info(
            f"Selected model: {best_model['provider']}/{best_model['model']} "
            f"(score: {best_model['score']:.2f})"
        )
        
        return best_model["provider"], best_model["model"]
    
    async def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: str = "general",
        **kwargs
    ) -> str:
        """
        Generate text using the best available model.
        
        Args:
            prompt: Input prompt
            provider: Specific provider to use
            model: Specific model to use
            task_type: Type of task for model selection
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if provider and model:
            # Use specific provider and model
            if provider not in self.clients:
                raise LLMError(f"Provider {provider} not available")
            
            client = self.clients[provider]
            return await client.generate_text(prompt, model, **kwargs)
        
        # Auto-select best model
        provider, model = await self.select_best_model(task_type=task_type)
        client = self.clients[provider]
        
        return await client.generate_text(prompt, model, **kwargs)
    
    async def analyze_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze image using vision-capable model.
        
        Args:
            image: PIL Image or base64 string
            prompt: Analysis prompt
            provider: Specific provider to use
            model: Specific model to use
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        if provider and model:
            # Use specific provider and model
            if provider not in self.clients:
                raise LLMError(f"Provider {provider} not available")
            
            client = self.clients[provider]
            return await client.analyze_image(image, prompt, model, **kwargs)
        
        # Auto-select best vision model
        provider, model = await self.select_best_model(
            task_type="vision",
            capabilities=["vision", "multimodal"]
        )
        client = self.clients[provider]
        
        return await client.analyze_image(image, prompt, model, **kwargs)
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        health_status = {}
        
        for provider_name, client in self.clients.items():
            try:
                is_healthy = await client.health_check()
                health_status[provider_name] = is_healthy
                self.model_health[provider_name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
                health_status[provider_name] = False
                self.model_health[provider_name] = False
        
        return health_status
    
    def _calculate_model_score(
        self,
        provider: str,
        model: Dict[str, Any],
        task_type: str,
        prefer_local: bool
    ) -> float:
        """
        Calculate a score for model selection.
        
        Args:
            provider: Provider name
            model: Model information
            task_type: Task type
            prefer_local: Prefer local models
            
        Returns:
            Model score (higher is better)
        """
        score = 0.0
        
        # Base score
        score += 1.0
        
        # Local preference
        if prefer_local and provider == "ollama":
            score += 2.0
        
        # Task-specific scoring
        capabilities = model.get("capabilities", [])
        
        if task_type == "vision" and "vision" in capabilities:
            score += 3.0
        elif task_type == "coding" and "coding" in capabilities:
            score += 2.0
        elif task_type == "reasoning" and "reasoning" in capabilities:
            score += 2.0
        
        # Cost consideration (lower cost = higher score)
        cost = model.get("cost_per_1k_tokens", 0)
        if cost == 0:  # Free models get bonus
            score += 1.0
        else:
            score += max(0, 1.0 - cost * 100)  # Penalty for expensive models
        
        # Context length bonus
        context_length = model.get("context_length", 4096)
        if context_length > 32000:
            score += 1.0
        elif context_length > 8000:
            score += 0.5
        
        return score
    
    def get_provider_client(self, provider: str) -> Optional[BaseLLMClient]:
        """
        Get client for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Client instance or None if not found
        """
        return self.clients.get(provider)
    
    def list_providers(self) -> List[str]:
        """
        Get list of available providers.
        
        Returns:
            List of provider names
        """
        return list(self.clients.keys())
    
    async def close(self):
        """Close all client connections."""
        for client in self.clients.values():
            if hasattr(client, "close"):
                await client.close()
