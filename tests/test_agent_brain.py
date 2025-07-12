"""Tests for the main agent brain."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from neura_orbit_agent.core.agent_brain import NeuraOrbitAgent, TaskResult
from neura_orbit_agent.utils.config import Config


class TestNeuraOrbitAgent:
    """Test cases for NeuraOrbitAgent class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Config.load_default()
        # Set test-friendly values
        config.agent.execution.max_retries = 1
        config.agent.execution.retry_delay = 0.1
        return config
    
    @pytest.fixture
    def agent(self, mock_config):
        """Create an agent instance for testing."""
        with patch('neura_orbit_agent.core.agent_brain.ScreenCapture'), \
             patch('neura_orbit_agent.core.agent_brain.LLMManager'), \
             patch('neura_orbit_agent.core.agent_brain.SystemController'), \
             patch('neura_orbit_agent.core.agent_brain.BrowserAutomation'):
            return NeuraOrbitAgent(mock_config)
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert agent.screen_capture is not None
        assert agent.llm_manager is not None
        assert agent.system_controller is not None
        assert agent.browser_automation is not None
        assert not agent.is_monitoring
        assert len(agent.task_history) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_screen(self, agent):
        """Test screen analysis functionality."""
        # Mock the screen capture and LLM response
        mock_image = MagicMock()
        agent.screen_capture.capture_screenshot.return_value = mock_image
        agent.llm_manager.analyze_image = AsyncMock(return_value="Test analysis result")
        agent.system_controller.get_system_info = AsyncMock(return_value={"platform": "test"})
        agent.system_controller.get_active_window = AsyncMock(return_value={"title": "Test Window"})
        
        result = await agent.analyze_screen()
        
        assert isinstance(result, str)
        assert len(result) > 0
        agent.llm_manager.analyze_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, agent):
        """Test successful task execution."""
        # Mock dependencies
        agent.analyze_screen = AsyncMock(return_value="Screen analysis")
        agent._create_execution_plan = AsyncMock(return_value={
            "steps": [
                {
                    "action": "analyze",
                    "description": "Test step",
                    "parameters": {},
                    "expected_outcome": "Success"
                }
            ]
        })
        agent._execute_plan = AsyncMock(return_value=TaskResult(True, "Success"))
        
        result = await agent.execute_task("Test task")
        
        assert isinstance(result, TaskResult)
        assert result.success
        assert len(agent.task_history) == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, agent):
        """Test failed task execution."""
        # Mock dependencies to simulate failure
        agent.analyze_screen = AsyncMock(side_effect=Exception("Test error"))
        
        result = await agent.execute_task("Test task")
        
        assert isinstance(result, TaskResult)
        assert not result.success
        assert "Test error" in result.message
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, agent):
        """Test monitoring start and stop."""
        # Mock analyze_screen to avoid actual screen capture
        agent.analyze_screen = AsyncMock(return_value="Monitor analysis")
        
        # Start monitoring
        await agent.start_monitoring(interval=0.1)
        assert agent.is_monitoring
        assert agent.monitoring_task is not None
        
        # Stop monitoring
        await agent.stop_monitoring()
        assert not agent.is_monitoring
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test health check functionality."""
        # Mock health check responses
        agent.llm_manager.health_check = AsyncMock(return_value={"ollama": True})
        agent.screen_capture.capture_screenshot = MagicMock()
        agent.system_controller.get_system_info = AsyncMock(return_value={})
        
        health = await agent.health_check()
        
        assert isinstance(health, dict)
        assert "llm_providers" in health
        assert "screen_capture" in health
        assert "system_controller" in health
        assert "overall" in health
    
    @pytest.mark.asyncio
    async def test_execute_step_click(self, agent):
        """Test executing a click step."""
        agent.system_controller.click = AsyncMock(return_value=True)
        
        result = await agent._execute_step("click", {"x": 100, "y": 200})
        
        assert result is True
        agent.system_controller.click.assert_called_once_with(100, 200)
    
    @pytest.mark.asyncio
    async def test_execute_step_type(self, agent):
        """Test executing a type step."""
        agent.system_controller.type_text = AsyncMock(return_value=True)
        
        result = await agent._execute_step("type", {"text": "Hello World"})
        
        assert result is True
        agent.system_controller.type_text.assert_called_once_with("Hello World")
    
    @pytest.mark.asyncio
    async def test_execute_step_unknown_action(self, agent):
        """Test executing an unknown action."""
        result = await agent._execute_step("unknown_action", {})
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_task_history(self, agent):
        """Test task history functionality."""
        # Initially empty
        history = await agent.get_task_history()
        assert len(history) == 0
        
        # Add a task to history manually
        agent.task_history.append({
            "description": "Test task",
            "result": True,
            "message": "Success",
            "execution_time": 1.0,
            "timestamp": 123456789
        })
        
        history = await agent.get_task_history()
        assert len(history) == 1
        assert history[0]["description"] == "Test task"
    
    @pytest.mark.asyncio
    async def test_close(self, agent):
        """Test agent cleanup."""
        agent.llm_manager.close = AsyncMock()
        agent.browser_automation.close_browser = AsyncMock()
        
        await agent.close()
        
        agent.llm_manager.close.assert_called_once()
        agent.browser_automation.close_browser.assert_called_once()


class TestTaskResult:
    """Test cases for TaskResult class."""
    
    def test_task_result_creation(self):
        """Test TaskResult creation."""
        result = TaskResult(True, "Success message", {"key": "value"}, 1.5)
        
        assert result.success is True
        assert result.message == "Success message"
        assert result.data == {"key": "value"}
        assert result.execution_time == 1.5
        assert result.timestamp > 0
    
    def test_task_result_defaults(self):
        """Test TaskResult with default values."""
        result = TaskResult(False, "Error message")
        
        assert result.success is False
        assert result.message == "Error message"
        assert result.data == {}
        assert result.execution_time == 0.0
        assert result.timestamp > 0
