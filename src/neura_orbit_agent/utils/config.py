"""Configuration management for Neura-Orbit-Agent."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    organization: Optional[str] = None
    timeout: int = 60
    models: List[Dict[str, Any]] = Field(default_factory=list)


class LLMConfig(BaseModel):
    """LLM configuration."""
    default_provider: str = "ollama"
    selection_strategy: str = "auto"
    providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)


class ScreenConfig(BaseModel):
    """Screen capture configuration."""
    capture_interval: float = 1.0
    monitor: int = 0
    resolution: str = "auto"
    compression_quality: int = 85
    format: str = "PNG"
    max_fps: int = 30
    buffer_size: int = 10
    blur_sensitive_areas: bool = False
    exclude_regions: List[Dict[str, int]] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    """Security configuration."""
    require_confirmation: bool = True
    confirmation_timeout: int = 30
    permissions: Dict[str, List[str]] = Field(default_factory=dict)
    audit_log: Dict[str, Any] = Field(default_factory=dict)


class AutomationConfig(BaseModel):
    """Automation configuration."""
    browser: Dict[str, Any] = Field(default_factory=dict)
    applications: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    learning: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    auth: Dict[str, Any] = Field(default_factory=dict)
    rate_limit: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: Dict[str, Any] = Field(default_factory=dict)
    console: Dict[str, Any] = Field(default_factory=dict)
    components: Dict[str, str] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    memory: Dict[str, Any] = Field(default_factory=dict)
    processing: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)


class DevelopmentConfig(BaseModel):
    """Development configuration."""
    debug: bool = False
    profiling: bool = False
    hot_reload: bool = False
    test_mode: bool = False
    mock_llm_responses: bool = False
    tools: Dict[str, Any] = Field(default_factory=dict)


class Config(BaseSettings):
    """Main configuration class."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields
        
    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Config instance
            
        Raises:
            ConfigurationError: If the configuration file is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
                
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    @classmethod
    def load_default(cls) -> "Config":
        """
        Load configuration with default values and environment variables.
        
        Returns:
            Config instance with defaults
        """
        return cls()
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.dict(),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def get_llm_provider_config(self, provider: str) -> Optional[LLMProviderConfig]:
        """
        Get configuration for a specific LLM provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider configuration or None if not found
        """
        return self.llm.providers.get(provider)
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the configuration and return any issues.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check LLM provider configurations
        for provider_name, provider_config in self.llm.providers.items():
            if provider_name in ["openai", "anthropic"] and not provider_config.api_key:
                issues.append(f"API key missing for {provider_name}")
        
        # Check required directories
        log_file = self.logging.file.get("path")
        if log_file:
            log_dir = Path(log_file).parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create log directory {log_dir}: {e}")
        
        return issues


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Config instance
    """
    if config_path:
        return Config.load_from_file(config_path)
    
    # Try to load from default locations
    default_paths = [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path.home() / ".neura-orbit" / "config.yaml",
    ]
    
    for path in default_paths:
        if path.exists():
            return Config.load_from_file(path)
    
    # Fall back to defaults with environment variables
    return Config.load_default()


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Config instance to set as global
    """
    global _config
    _config = config
