"""Main agent brain for decision making and task execution."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .screen_capture import ScreenCapture
from .llm_manager import LLMManager
from .system_controller import SystemController
from ..automation.browser_automation import BrowserAutomation
from ..utils.config import Config
from ..utils.exceptions import NeuraOrbitError, TaskExecutionError
from ..utils.logger import get_logger

logger = get_logger("neura_orbit.agent")


class TaskResult:
    """Result of a task execution."""
    
    def __init__(
        self,
        success: bool,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0
    ):
        self.success = success
        self.message = message
        self.data = data or {}
        self.execution_time = execution_time
        self.timestamp = time.time()


class NeuraOrbitAgent:
    """Main agent for screen monitoring and task execution."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Neura-Orbit Agent.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        
        # Initialize core components
        self.screen_capture = ScreenCapture(self.config)
        self.llm_manager = LLMManager(self.config)
        self.system_controller = SystemController(self.config)
        self.browser_automation = BrowserAutomation(self.config)
        
        # Agent state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.task_history: List[Dict[str, Any]] = []
        
        # Reasoning and execution settings
        self.agent_config = self.config.agent
        self.max_retries = self.agent_config.execution.get("max_retries", 3)
        self.retry_delay = self.agent_config.execution.get("retry_delay", 2.0)
        
        logger.info("Neura-Orbit Agent initialized")
    
    async def analyze_screen(
        self,
        prompt: Optional[str] = None,
        include_system_info: bool = True
    ) -> str:
        """
        Analyze the current screen content.
        
        Args:
            prompt: Custom analysis prompt
            include_system_info: Include system information in analysis
            
        Returns:
            Analysis result
        """
        try:
            # Capture screenshot
            screenshot = self.screen_capture.capture_screenshot()
            
            # Default analysis prompt
            if prompt is None:
                prompt = """
                Analyze this screenshot and describe:
                1. What application or website is currently active
                2. What the user might be trying to do
                3. Any notable elements, errors, or opportunities for automation
                4. Suggested next actions
                
                Be concise but thorough in your analysis.
                """
            
            # Add system context if requested
            if include_system_info:
                system_info = await self.system_controller.get_system_info()
                active_window = await self.system_controller.get_active_window()
                
                context = f"""
                System Context:
                - Platform: {system_info.get('platform', 'unknown')}
                - CPU Usage: {system_info.get('cpu_percent', 0):.1f}%
                - Memory Usage: {system_info.get('memory_percent', 0):.1f}%
                - Active Window: {active_window.get('title', 'Unknown') if active_window else 'Unknown'}
                
                {prompt}
                """
                prompt = context
            
            # Analyze with LLM
            analysis = await self.llm_manager.analyze_image(
                screenshot,
                prompt,
                task_type="vision"
            )
            
            logger.info("Screen analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Screen analysis failed: {e}")
            return f"Failed to analyze screen: {e}"
    
    async def execute_task(self, task_description: str) -> TaskResult:
        """
        Execute a natural language task.
        
        Args:
            task_description: Description of task to execute
            
        Returns:
            TaskResult object
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing task: {task_description}")
            
            # Analyze current screen state
            screen_analysis = await self.analyze_screen(
                prompt=f"""
                The user wants to: {task_description}
                
                Analyze the current screen and determine:
                1. What steps are needed to complete this task
                2. What applications or websites need to be accessed
                3. Any obstacles or requirements
                4. A step-by-step action plan
                """
            )
            
            # Plan the task execution
            execution_plan = await self._create_execution_plan(
                task_description,
                screen_analysis
            )
            
            # Execute the plan
            result = await self._execute_plan(execution_plan)
            
            execution_time = time.time() - start_time
            
            # Record task in history
            task_record = {
                "description": task_description,
                "result": result.success,
                "message": result.message,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            self.task_history.append(task_record)
            
            logger.info(f"Task completed in {execution_time:.2f}s: {result.message}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {e}"
            logger.error(error_msg)
            
            return TaskResult(
                success=False,
                message=error_msg,
                execution_time=execution_time
            )
    
    async def start_monitoring(
        self,
        interval: float = 5.0,
        callback: Optional[callable] = None
    ) -> None:
        """
        Start continuous screen monitoring.
        
        Args:
            interval: Monitoring interval in seconds
            callback: Optional callback for monitoring events
        """
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        logger.info(f"Starting screen monitoring (interval: {interval}s)")
        
        async def monitor_loop():
            while self.is_monitoring:
                try:
                    # Analyze current screen
                    analysis = await self.analyze_screen(
                        prompt="Monitor this screen for any changes, errors, or opportunities for automation. Be brief."
                    )
                    
                    # Call callback if provided
                    if callback:
                        await callback({
                            "timestamp": time.time(),
                            "analysis": analysis,
                            "screenshot": self.screen_capture.get_latest_frame()
                        })
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(interval)
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop screen monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Screen monitoring stopped")
    
    async def _create_execution_plan(
        self,
        task_description: str,
        screen_analysis: str
    ) -> Dict[str, Any]:
        """
        Create an execution plan for a task.
        
        Args:
            task_description: Task description
            screen_analysis: Current screen analysis
            
        Returns:
            Execution plan dictionary
        """
        planning_prompt = f"""
        Task: {task_description}
        Current Screen Analysis: {screen_analysis}
        
        Create a detailed execution plan as a JSON object with this structure:
        {{
            "steps": [
                {{
                    "action": "action_type",
                    "description": "what to do",
                    "parameters": {{"param1": "value1"}},
                    "expected_outcome": "what should happen"
                }}
            ],
            "requirements": ["list of requirements"],
            "estimated_time": "time estimate",
            "risk_level": "low/medium/high"
        }}
        
        Available actions:
        - "click": Click at coordinates or element
        - "type": Type text
        - "key_press": Press keyboard keys
        - "open_app": Open application
        - "navigate_browser": Navigate to URL
        - "wait": Wait for condition
        - "screenshot": Take screenshot
        - "analyze": Analyze screen
        
        Be specific and actionable. Only include steps that are necessary and safe.
        """
        
        try:
            plan_response = await self.llm_manager.generate_text(
                planning_prompt,
                task_type="reasoning"
            )
            
            # Try to parse as JSON
            plan = json.loads(plan_response)
            logger.debug(f"Created execution plan with {len(plan.get('steps', []))} steps")
            return plan
            
        except json.JSONDecodeError:
            # Fallback to simple plan
            logger.warning("Failed to parse execution plan as JSON, using fallback")
            return {
                "steps": [
                    {
                        "action": "analyze",
                        "description": "Analyze current situation",
                        "parameters": {},
                        "expected_outcome": "Understanding of current state"
                    }
                ],
                "requirements": ["Screen access"],
                "estimated_time": "30 seconds",
                "risk_level": "low"
            }
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> TaskResult:
        """
        Execute a task plan.
        
        Args:
            plan: Execution plan
            
        Returns:
            TaskResult
        """
        steps = plan.get("steps", [])
        if not steps:
            return TaskResult(False, "No steps in execution plan")
        
        executed_steps = 0
        
        for i, step in enumerate(steps):
            action = step.get("action")
            description = step.get("description", "")
            parameters = step.get("parameters", {})
            
            logger.info(f"Executing step {i+1}/{len(steps)}: {description}")
            
            try:
                success = await self._execute_step(action, parameters)
                
                if not success:
                    return TaskResult(
                        False,
                        f"Step {i+1} failed: {description}",
                        {"completed_steps": executed_steps}
                    )
                
                executed_steps += 1
                
                # Small delay between steps
                await asyncio.sleep(0.5)
                
            except Exception as e:
                return TaskResult(
                    False,
                    f"Error in step {i+1}: {e}",
                    {"completed_steps": executed_steps}
                )
        
        return TaskResult(
            True,
            f"Successfully completed all {executed_steps} steps",
            {"completed_steps": executed_steps}
        )
    
    async def _execute_step(self, action: str, parameters: Dict[str, Any]) -> bool:
        """
        Execute a single step.
        
        Args:
            action: Action type
            parameters: Action parameters
            
        Returns:
            True if successful
        """
        try:
            if action == "click":
                x = parameters.get("x", 0)
                y = parameters.get("y", 0)
                return await self.system_controller.click(x, y)
            
            elif action == "type":
                text = parameters.get("text", "")
                return await self.system_controller.type_text(text)
            
            elif action == "key_press":
                key = parameters.get("key", "")
                return await self.system_controller.press_key(key)
            
            elif action == "open_app":
                app_name = parameters.get("app_name", "")
                return await self.system_controller.open_application(app_name)
            
            elif action == "navigate_browser":
                url = parameters.get("url", "")
                await self.browser_automation.start_browser()
                return await self.browser_automation.navigate_to(url)
            
            elif action == "wait":
                duration = parameters.get("duration", 1.0)
                await asyncio.sleep(duration)
                return True
            
            elif action == "screenshot":
                self.screen_capture.capture_screenshot()
                return True
            
            elif action == "analyze":
                await self.analyze_screen()
                return True
            
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Step execution error: {e}")
            return False
    
    async def get_task_history(self) -> List[Dict[str, Any]]:
        """
        Get task execution history.
        
        Returns:
            List of task records
        """
        return self.task_history.copy()
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            Health status of components
        """
        health = {}
        
        try:
            # Check LLM providers
            llm_health = await self.llm_manager.health_check()
            health["llm_providers"] = llm_health
            
            # Check screen capture
            try:
                self.screen_capture.capture_screenshot()
                health["screen_capture"] = True
            except Exception:
                health["screen_capture"] = False
            
            # Check system controller
            try:
                await self.system_controller.get_system_info()
                health["system_controller"] = True
            except Exception:
                health["system_controller"] = False
            
            health["overall"] = all(
                isinstance(v, bool) and v or 
                isinstance(v, dict) and any(v.values())
                for v in health.values()
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)
        
        return health
    
    async def close(self):
        """Close all components and cleanup."""
        await self.stop_monitoring()
        await self.llm_manager.close()
        await self.browser_automation.close_browser()
        logger.info("Agent closed")
