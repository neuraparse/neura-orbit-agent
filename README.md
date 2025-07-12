# 🚀 Neura-Orbit-Agent

**Advanced AI Agent for Screen Monitoring and System Automation**

Neura-Orbit-Agent is a powerful, cross-platform AI agent that can monitor your screen in real-time, understand what's happening, and perform automated tasks using various LLM models (Ollama, OpenAI, Anthropic). It can control your system across Linux, Windows, and macOS environments.

## ✨ Features

- 🖥️ **Real-time Screen Monitoring**: Continuous screen capture and analysis
- 🤖 **Multi-LLM Support**: Integration with Ollama, OpenAI, and Anthropic models
- 🔄 **Intelligent Model Selection**: Automatic model selection based on task requirements
- 🌐 **Cross-Platform**: Works on Linux, Windows, and macOS
- 🎯 **Agentic Automation**: Natural language task execution
- 🔒 **Security First**: Secure system access with permission management
- 🌐 **Web Automation**: Browser control and web interaction
- 📱 **Application Control**: Native application automation
- 🔧 **CLI & API**: Both command-line and REST API interfaces

## 🎯 Use Cases

```bash
# Natural language commands
noa "go to browser, navigate to neuraparse.com and run software tests"
noa "open VS Code, go to last project and push to GitHub"
noa "take a screenshot, analyze what's on screen and summarize"
noa "monitor screen for errors and alert me"
```

## 🏗️ Architecture

```
neura-orbit-agent/
├── core/                    # Core system components
│   ├── screen_capture.py    # Screen capture functionality
│   ├── llm_manager.py       # LLM model management
│   ├── system_controller.py # OS control
│   └── agent_brain.py       # Agentic decision making
├── integrations/            # External service integrations
├── automation/              # Automation modules
├── security/                # Security and permissions
├── cli/                     # Command line interface
└── api/                     # REST API
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Ollama (optional, for local models)
- OpenAI API key (optional)
- Anthropic API key (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/bayrameker/neura-orbit-agent.git
cd neura-orbit-agent

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration with your API keys and preferences
nano config/config.yaml
```

### Basic Usage

```bash
# Start the agent
noa start

# Execute a task
noa execute "take a screenshot and describe what you see"

# Monitor mode
noa monitor --interval 5

# Interactive mode
noa interactive
```

## 🔧 Configuration

The agent uses a YAML configuration file for settings:

```yaml
# config/config.yaml
llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      models: ["llama3.2", "qwen2.5"]
    openai:
      api_key: "${OPENAI_API_KEY}"
      models: ["gpt-4", "gpt-3.5-turbo"]
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      models: ["claude-3-sonnet", "claude-3-haiku"]

screen:
  capture_interval: 1.0
  resolution: "auto"
  monitor: 0

security:
  require_confirmation: true
  allowed_actions: ["screenshot", "read", "analyze"]
  restricted_actions: ["delete", "install", "network"]

automation:
  browser:
    default: "chrome"
    headless: false
  timeout: 30
```

## 🛡️ Security

Neura-Orbit-Agent takes security seriously:

- **Permission-based access**: All system actions require explicit permissions
- **Action confirmation**: Destructive actions require user confirmation
- **Audit logging**: All actions are logged for review
- **Sandboxed execution**: Isolated execution environment for safety

## 🧪 Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=neura_orbit_agent
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with verbose output
pytest -v
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Security Guide](docs/security.md)
- [Contributing](docs/contributing.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM support
- [OpenAI](https://openai.com/) for GPT models
- [Anthropic](https://anthropic.com/) for Claude models
- The open-source community for amazing tools and libraries

## 📞 Support

- 📧 Email: support@neura-orbit-agent.com
- 💬 Discord: [Join our community](https://discord.gg/neura-orbit)
- 🐛 Issues: [GitHub Issues](https://github.com/bayrameker/neura-orbit-agent/issues)

---

**⚠️ Disclaimer**: This tool can perform system-level operations. Always review and understand the actions before execution. Use responsibly and in accordance with your organization's security policies.
