"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from neura_orbit_agent.utils.config import Config
from neura_orbit_agent.utils.exceptions import ConfigurationError


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        config = Config.load_default()
        
        assert config is not None
        assert config.llm is not None
        assert config.screen is not None
        assert config.security is not None
        assert config.automation is not None
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config.load_default()
        issues = config.validate_configuration()
        
        # Should be a list (may be empty or contain issues)
        assert isinstance(issues, list)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration from file."""
        config = Config.load_default()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save config
            config.save_to_file(temp_path)
            assert temp_path.exists()
            
            # Load config
            loaded_config = Config.load_from_file(temp_path)
            
            # Compare key attributes
            assert loaded_config.llm.default_provider == config.llm.default_provider
            assert loaded_config.screen.capture_interval == config.screen.capture_interval
            
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    def test_invalid_config_file(self):
        """Test loading invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError):
                Config.load_from_file(temp_path)
        finally:
            temp_path.unlink()
    
    def test_nonexistent_config_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(ConfigurationError):
            Config.load_from_file("nonexistent_file.yaml")
    
    def test_llm_provider_config(self):
        """Test LLM provider configuration access."""
        config = Config.load_default()
        
        # Test getting provider config
        ollama_config = config.get_llm_provider_config("ollama")
        assert ollama_config is not None
        
        # Test non-existent provider
        nonexistent_config = config.get_llm_provider_config("nonexistent")
        assert nonexistent_config is None
