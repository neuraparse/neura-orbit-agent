"""
2025 Intelligent Task Planning and Execution System
Uses LangGraph for state management and multi-agent coordination
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..vision.intelligent_vision import IntelligentVision, ScreenAnalysis, UIElement
from ..core.llm_manager import LLMManager
from ..core.system_controller import SystemController
from ..automation.browser_automation import BrowserAutomation
from ..utils.config import Config
from ..utils.exceptions import NeuraOrbitError, TaskExecutionError
from ..utils.logger import get_logger

logger = get_logger("neura_orbit.planner")


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    BLOCKED = "blocked"


class ActionType(Enum):
    """Types of actions the system can perform."""
    CLICK = "click"
    TYPE = "type"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    WAIT = "wait"
    VERIFY = "verify"
    NAVIGATE = "navigate"
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    SCREENSHOT = "screenshot"
    ANALYZE = "analyze"
    RECOVER = "recover"


@dataclass
class Action:
    """Individual action with intelligent properties."""
    action_type: ActionType
    parameters: Dict[str, Any]
    description: str
    expected_outcome: str
    confidence: float
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    verification_method: Optional[str] = None
    fallback_actions: List['Action'] = field(default_factory=list)


@dataclass
class TaskPlan:
    """Intelligent task execution plan."""
    task_id: str
    description: str
    actions: List[Action]
    success_criteria: List[str]
    failure_indicators: List[str]
    estimated_duration: float
    complexity_score: float
    risk_level: str
    created_at: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    """Current execution state with context."""
    current_action_index: int = 0
    status: TaskStatus = TaskStatus.PENDING
    last_screenshot: Optional[ScreenAnalysis] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0
    start_time: Optional[datetime] = None
    context_memory: Dict[str, Any] = field(default_factory=dict)


class IntelligentPlanner:
    """
    2025 Advanced Task Planning and Execution System.
    
    Features:
    - Intelligent task decomposition
    - Adaptive execution with error recovery
    - Context-aware decision making
    - Self-healing capabilities
    - Multi-modal verification
    - Learning from failures
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the intelligent planner."""
        self.config = config or Config.load_default()
        
        # Core components
        self.llm_manager = LLMManager(self.config)
        self.vision_system = IntelligentVision(self.config)
        self.system_controller = SystemController(self.config)
        self.browser_automation = BrowserAutomation(self.config)
        
        # Execution state
        self.current_plan: Optional[TaskPlan] = None
        self.execution_state = ExecutionState()
        
        # Learning and adaptation
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        logger.info("Intelligent Planner initialized with 2025 capabilities")
    
    async def plan_and_execute_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plan and execute a task with intelligent adaptation.
        
        Args:
            task_description: Natural language task description
            context: Additional context for task execution
            
        Returns:
            Execution result with detailed metrics
        """
        try:
            logger.info(f"Starting intelligent task planning: {task_description}")
            
            # Phase 1: Initial Analysis and Planning
            self.execution_state = ExecutionState(
                status=TaskStatus.PLANNING,
                start_time=datetime.now(),
                context_memory=context or {}
            )
            
            # Analyze current screen state
            current_screen = await self._capture_and_analyze_screen()
            self.execution_state.last_screenshot = current_screen
            
            # Create intelligent task plan
            task_plan = await self._create_intelligent_plan(
                task_description, current_screen, context
            )
            self.current_plan = task_plan
            
            logger.info(f"Task plan created with {len(task_plan.actions)} actions")
            
            # Phase 2: Adaptive Execution
            self.execution_state.status = TaskStatus.EXECUTING
            execution_result = await self._execute_plan_adaptively(task_plan)
            
            # Phase 3: Learning and Metrics
            await self._learn_from_execution(task_plan, execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Task planning and execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - (self.execution_state.start_time.timestamp() if self.execution_state.start_time else time.time()),
                "actions_completed": self.execution_state.current_action_index
            }
    
    async def _create_intelligent_plan(
        self,
        task_description: str,
        current_screen: ScreenAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> TaskPlan:
        """Create an intelligent, adaptive task plan."""
        
        planning_prompt = f"""
        You are an expert AI task planner for computer automation. Create a detailed, intelligent plan.
        
        TASK: {task_description}
        
        CURRENT SCREEN STATE:
        - Layout: {current_screen.layout_description}
        - Current State: {current_screen.current_state}
        - Available Elements: {len(current_screen.elements)} interactive elements
        - Detected Errors: {current_screen.errors_detected}
        - Possible Actions: {current_screen.possible_actions}
        
        CONTEXT: {json.dumps(context or {}, indent=2)}
        
        Create a step-by-step plan with:
        
        1. TASK DECOMPOSITION:
           - Break down into atomic, verifiable actions
           - Each action should have clear success criteria
           - Include verification steps
           - Plan for error scenarios
        
        2. INTELLIGENT SEQUENCING:
           - Optimal order of operations
           - Dependencies between actions
           - Parallel execution opportunities
           - Checkpoints for verification
        
        3. ADAPTIVE STRATEGIES:
           - Alternative approaches for each step
           - Error recovery mechanisms
           - Timeout handling
           - Fallback plans
        
        4. SUCCESS CRITERIA:
           - How to verify task completion
           - Intermediate success indicators
           - Final validation methods
        
        5. RISK ASSESSMENT:
           - Potential failure points
           - Complexity estimation
           - Time estimation
           - Risk mitigation strategies
        
        Return as structured JSON with detailed action specifications.
        """
        
        try:
            # Use advanced reasoning model for planning
            planning_response = await self.llm_manager.generate_text(
                planning_prompt,
                task_type="reasoning",
                max_tokens=4000
            )
            
            # Parse and structure the plan
            plan_data = self._parse_planning_response(planning_response)
            
            # Create TaskPlan object
            task_plan = TaskPlan(
                task_id=f"task_{int(time.time())}",
                description=task_description,
                actions=plan_data["actions"],
                success_criteria=plan_data["success_criteria"],
                failure_indicators=plan_data["failure_indicators"],
                estimated_duration=plan_data["estimated_duration"],
                complexity_score=plan_data["complexity_score"],
                risk_level=plan_data["risk_level"],
                created_at=datetime.now(),
                context=context or {}
            )
            
            return task_plan
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Create fallback simple plan
            return self._create_fallback_plan(task_description, current_screen)
    
    async def _execute_plan_adaptively(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute plan with intelligent adaptation and error recovery."""
        
        execution_log = []
        start_time = time.time()
        
        try:
            for i, action in enumerate(plan.actions):
                self.execution_state.current_action_index = i
                
                logger.info(f"Executing action {i+1}/{len(plan.actions)}: {action.description}")
                
                # Pre-action analysis
                pre_screen = await self._capture_and_analyze_screen()
                
                # Execute action with intelligent retry
                action_result = await self._execute_action_intelligently(action, pre_screen)
                
                # Post-action verification
                post_screen = await self._capture_and_analyze_screen()
                verification_result = await self._verify_action_success(
                    action, pre_screen, post_screen, action_result
                )
                
                # Log execution step
                step_log = {
                    "action_index": i,
                    "action": action.description,
                    "result": action_result,
                    "verification": verification_result,
                    "timestamp": time.time(),
                    "screen_state": post_screen.current_state
                }
                execution_log.append(step_log)
                self.execution_state.execution_log.append(step_log)
                
                # Handle verification results
                if not verification_result["success"]:
                    if action.retry_count < action.max_retries:
                        logger.warning(f"Action failed, retrying: {verification_result['reason']}")
                        action.retry_count += 1
                        i -= 1  # Retry same action
                        continue
                    else:
                        # Try fallback actions
                        if action.fallback_actions:
                            logger.info("Trying fallback actions")
                            for fallback in action.fallback_actions:
                                fallback_result = await self._execute_action_intelligently(
                                    fallback, post_screen
                                )
                                if fallback_result["success"]:
                                    break
                        else:
                            # Critical failure
                            logger.error(f"Action failed permanently: {action.description}")
                            self.execution_state.status = TaskStatus.FAILED
                            break
                
                # Check for task completion
                if await self._check_task_completion(plan, post_screen):
                    logger.info("Task completed successfully!")
                    self.execution_state.status = TaskStatus.COMPLETED
                    break
                
                # Adaptive delay based on action type and system state
                await self._adaptive_delay(action, post_screen)
            
            # Final verification
            final_screen = await self._capture_and_analyze_screen()
            final_success = await self._verify_task_completion(plan, final_screen)
            
            execution_time = time.time() - start_time
            
            return {
                "success": final_success,
                "execution_time": execution_time,
                "actions_completed": self.execution_state.current_action_index + 1,
                "total_actions": len(plan.actions),
                "execution_log": execution_log,
                "final_state": final_screen.current_state,
                "performance_metrics": self._calculate_performance_metrics(plan, execution_log)
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "actions_completed": self.execution_state.current_action_index,
                "execution_log": execution_log
            }
    
    async def _execute_action_intelligently(
        self,
        action: Action,
        screen_context: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Execute a single action with intelligent adaptation."""
        
        try:
            logger.debug(f"Executing {action.action_type.value}: {action.description}")
            
            if action.action_type == ActionType.CLICK:
                return await self._execute_click_action(action, screen_context)
            elif action.action_type == ActionType.TYPE:
                return await self._execute_type_action(action, screen_context)
            elif action.action_type == ActionType.KEY_PRESS:
                return await self._execute_key_action(action, screen_context)
            elif action.action_type == ActionType.WAIT:
                return await self._execute_wait_action(action, screen_context)
            elif action.action_type == ActionType.VERIFY:
                return await self._execute_verify_action(action, screen_context)
            elif action.action_type == ActionType.NAVIGATE:
                return await self._execute_navigate_action(action, screen_context)
            elif action.action_type == ActionType.OPEN_APP:
                return await self._execute_open_app_action(action, screen_context)
            else:
                return {"success": False, "error": f"Unknown action type: {action.action_type}"}
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_click_action(
        self,
        action: Action,
        screen_context: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Execute click action with intelligent element detection."""
        
        # Find target element intelligently
        target_element = await self._find_target_element(
            action.parameters.get("target_description", ""),
            screen_context
        )
        
        if not target_element:
            return {"success": False, "error": "Target element not found"}
        
        # Calculate optimal click coordinates
        click_x = target_element.bbox[0] + target_element.bbox[2] // 2
        click_y = target_element.bbox[1] + target_element.bbox[3] // 2
        
        # Execute click with system controller
        success = await self.system_controller.click(
            click_x, click_y,
            button=action.parameters.get("button", "left"),
            confirm=False
        )
        
        return {
            "success": success,
            "coordinates": (click_x, click_y),
            "element": target_element.description
        }
    
    async def _execute_type_action(
        self,
        action: Action,
        screen_context: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Execute typing action with intelligent text handling."""
        
        text = action.parameters.get("text", "")
        clear_first = action.parameters.get("clear_first", False)
        
        if clear_first:
            # Select all and delete
            await self.system_controller.press_key(["cmd", "a"], confirm=False)
            await asyncio.sleep(0.1)
            await self.system_controller.press_key("backspace", confirm=False)
            await asyncio.sleep(0.1)
        
        success = await self.system_controller.type_text(text, confirm=False)
        
        return {
            "success": success,
            "text": text,
            "cleared_first": clear_first
        }
    
    async def _execute_key_action(
        self,
        action: Action,
        screen_context: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Execute key press action."""
        
        keys = action.parameters.get("keys", [])
        if isinstance(keys, str):
            keys = [keys]
        
        success = await self.system_controller.press_key(keys, confirm=False)
        
        return {
            "success": success,
            "keys": keys
        }
    
    async def _execute_wait_action(
        self,
        action: Action,
        screen_context: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Execute intelligent wait with condition checking."""
        
        duration = action.parameters.get("duration", 1.0)
        condition = action.parameters.get("condition")
        
        if condition:
            # Wait for specific condition
            max_wait = action.parameters.get("max_wait", 30.0)
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                current_screen = await self._capture_and_analyze_screen()
                if await self._check_condition(condition, current_screen):
                    return {"success": True, "condition_met": True, "wait_time": time.time() - start_time}
                await asyncio.sleep(0.5)
            
            return {"success": False, "condition_met": False, "timeout": True}
        else:
            # Simple duration wait
            await asyncio.sleep(duration)
            return {"success": True, "duration": duration}
    
    async def _capture_and_analyze_screen(self) -> ScreenAnalysis:
        """Capture and analyze current screen state."""
        from ..core.screen_capture import ScreenCapture
        
        screen_capture = ScreenCapture(self.config)
        screenshot = screen_capture.capture_screenshot()
        
        analysis = await self.vision_system.analyze_screen_intelligent(
            screenshot,
            task_context=self.current_plan.description if self.current_plan else None,
            previous_action=self._get_last_action_description()
        )
        
        return analysis
    
    def _get_last_action_description(self) -> Optional[str]:
        """Get description of last executed action."""
        if self.execution_state.execution_log:
            return self.execution_state.execution_log[-1]["action"]
        return None
    
    # Additional helper methods would be implemented here...
    
    def _parse_planning_response(self, response: str) -> Dict[str, Any]:
        """Parse planning response into structured data."""
        # Implementation would parse the LLM response
        return {
            "actions": [],
            "success_criteria": [],
            "failure_indicators": [],
            "estimated_duration": 30.0,
            "complexity_score": 0.5,
            "risk_level": "medium"
        }
    
    def _create_fallback_plan(self, task_description: str, screen: ScreenAnalysis) -> TaskPlan:
        """Create a simple fallback plan."""
        return TaskPlan(
            task_id=f"fallback_{int(time.time())}",
            description=task_description,
            actions=[],
            success_criteria=["Task completed"],
            failure_indicators=["Error occurred"],
            estimated_duration=30.0,
            complexity_score=0.3,
            risk_level="low",
            created_at=datetime.now()
        )
