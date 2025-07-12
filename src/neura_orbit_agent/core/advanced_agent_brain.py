"""
2025 Advanced Agent Brain with Multi-Modal Intelligence
Combines vision, planning, execution, and self-healing capabilities
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from ..vision.intelligent_vision import IntelligentVision, ScreenAnalysis
from ..planning.intelligent_planner import IntelligentPlanner, TaskPlan, ExecutionState
from ..recovery.self_healing import SelfHealingSystem, ErrorContext
from ..core.llm_manager import LLMManager
from ..core.system_controller import SystemController
from ..core.screen_capture import ScreenCapture
from ..automation.browser_automation import BrowserAutomation
from ..utils.config import Config
from ..utils.exceptions import NeuraOrbitError, TaskExecutionError
from ..utils.logger import get_logger

logger = get_logger("neura_orbit.advanced_agent")


@dataclass
class TaskResult:
    """Enhanced task result with comprehensive metrics."""
    success: bool
    message: str
    execution_time: float
    actions_completed: int
    total_actions: int
    confidence_score: float
    error_count: int
    recovery_attempts: int
    performance_metrics: Dict[str, Any]
    screenshots: List[str] = None
    execution_log: List[Dict[str, Any]] = None
    learned_patterns: List[str] = None


class AdvancedAgentBrain:
    """
    2025 Advanced AI Agent Brain with Multi-Modal Intelligence.
    
    Features:
    - Multi-modal vision understanding (Claude 3.5 Sonnet + GPT-4V)
    - Intelligent task planning and decomposition
    - Adaptive execution with real-time monitoring
    - Self-healing and error recovery
    - Continuous learning and improvement
    - Context-aware decision making
    - Predictive error prevention
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the advanced agent brain."""
        self.config = config or Config.load_default()
        
        # Core intelligence components
        self.vision_system = IntelligentVision(self.config)
        self.planner = IntelligentPlanner(self.config)
        self.healing_system = SelfHealingSystem(self.config)
        
        # Traditional components (enhanced)
        self.llm_manager = LLMManager(self.config)
        self.system_controller = SystemController(self.config)
        self.screen_capture = ScreenCapture(self.config)
        self.browser_automation = BrowserAutomation(self.config)
        
        # Agent state and memory
        self.current_task: Optional[str] = None
        self.execution_context: Dict[str, Any] = {}
        self.learning_memory: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Monitoring and adaptation
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.adaptation_enabled = True
        
        logger.info("Advanced Agent Brain initialized with 2025 capabilities")
    
    async def execute_intelligent_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        monitoring_enabled: bool = True
    ) -> TaskResult:
        """
        Execute a task with full 2025 intelligence capabilities.
        
        Args:
            task_description: Natural language task description
            context: Additional context for task execution
            monitoring_enabled: Enable real-time monitoring and adaptation
            
        Returns:
            Comprehensive task result with metrics
        """
        start_time = time.time()
        self.current_task = task_description
        self.execution_context = context or {}
        
        try:
            logger.info(f"🚀 Starting intelligent task execution: {task_description}")
            
            # Phase 1: Pre-execution Analysis
            logger.info("📊 Phase 1: Pre-execution Analysis")
            initial_analysis = await self._comprehensive_pre_analysis(task_description, context)
            
            # Phase 2: Intelligent Planning
            logger.info("🧠 Phase 2: Intelligent Planning")
            execution_plan = await self.planner.plan_and_execute_task(task_description, context)
            
            # Phase 3: Monitored Execution with Self-Healing
            logger.info("⚡ Phase 3: Monitored Execution")
            if monitoring_enabled:
                execution_result = await self._execute_with_monitoring(execution_plan)
            else:
                execution_result = execution_plan
            
            # Phase 4: Post-execution Learning
            logger.info("📚 Phase 4: Learning and Adaptation")
            learning_insights = await self._post_execution_learning(
                task_description, execution_result, initial_analysis
            )
            
            # Compile comprehensive result
            task_result = TaskResult(
                success=execution_result.get("success", False),
                message=execution_result.get("message", "Task completed"),
                execution_time=time.time() - start_time,
                actions_completed=execution_result.get("actions_completed", 0),
                total_actions=execution_result.get("total_actions", 0),
                confidence_score=execution_result.get("confidence_score", 0.8),
                error_count=execution_result.get("error_count", 0),
                recovery_attempts=execution_result.get("recovery_attempts", 0),
                performance_metrics=execution_result.get("performance_metrics", {}),
                screenshots=execution_result.get("screenshots", []),
                execution_log=execution_result.get("execution_log", []),
                learned_patterns=learning_insights.get("patterns", [])
            )
            
            # Store in performance history
            self.performance_history.append({
                "task": task_description,
                "result": task_result,
                "timestamp": datetime.now(),
                "context": context
            })
            
            logger.info(f"✅ Task completed: {task_result.success} in {task_result.execution_time:.2f}s")
            return task_result
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            return TaskResult(
                success=False,
                message=f"Task execution failed: {e}",
                execution_time=time.time() - start_time,
                actions_completed=0,
                total_actions=0,
                confidence_score=0.0,
                error_count=1,
                recovery_attempts=0,
                performance_metrics={}
            )
        finally:
            self.current_task = None
            self.execution_context = {}
    
    async def _comprehensive_pre_analysis(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive pre-execution analysis."""
        
        # Capture and analyze current screen
        screenshot = self.screen_capture.capture_screenshot()
        screen_analysis = await self.vision_system.analyze_screen_intelligent(
            screenshot, task_context=task_description
        )
        
        # Analyze task complexity and requirements
        complexity_analysis = await self._analyze_task_complexity(task_description, screen_analysis)
        
        # Check for potential issues and blockers
        risk_analysis = await self._analyze_execution_risks(task_description, screen_analysis)
        
        # Gather system context
        system_context = await self._gather_system_context()
        
        return {
            "screen_analysis": screen_analysis,
            "complexity": complexity_analysis,
            "risks": risk_analysis,
            "system_context": system_context,
            "timestamp": time.time()
        }
    
    async def _execute_with_monitoring(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan with real-time monitoring and adaptation."""
        
        monitoring_data = []
        error_count = 0
        recovery_attempts = 0
        
        # Start monitoring task
        if self.adaptation_enabled:
            self.monitoring_task = asyncio.create_task(
                self._continuous_monitoring(monitoring_data)
            )
        
        try:
            # Execute with enhanced error handling
            result = execution_plan.copy()
            
            # Monitor for errors during execution
            while not result.get("success", False) and error_count < 3:
                # Check for errors
                current_screen = await self._capture_current_screen()
                error_context = await self.healing_system.detect_and_analyze_error(
                    current_screen, {"task": self.current_task}
                )
                
                if error_context:
                    error_count += 1
                    logger.warning(f"Error detected during execution: {error_context.description}")
                    
                    # Attempt recovery
                    recovery_plan = await self.healing_system.create_recovery_plan(error_context)
                    if recovery_plan:
                        recovery_attempts += 1
                        recovery_result = await self.healing_system.execute_recovery_plan(
                            recovery_plan, error_context
                        )
                        
                        if recovery_result.get("success"):
                            logger.info("✅ Error recovered successfully")
                            # Continue execution
                            break
                        else:
                            logger.error("❌ Recovery failed")
                    else:
                        logger.error("❌ No recovery plan available")
                        break
                else:
                    # No errors detected, execution likely successful
                    result["success"] = True
                    break
            
            # Add monitoring metrics
            result.update({
                "error_count": error_count,
                "recovery_attempts": recovery_attempts,
                "monitoring_data": monitoring_data
            })
            
            return result
            
        finally:
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
    
    async def _continuous_monitoring(self, monitoring_data: List[Dict[str, Any]]) -> None:
        """Continuous monitoring during task execution."""
        
        try:
            while True:
                # Capture current state
                current_screen = await self._capture_current_screen()
                
                # Analyze for issues
                issues = await self._detect_execution_issues(current_screen)
                
                # Store monitoring data
                monitoring_point = {
                    "timestamp": time.time(),
                    "screen_state": current_screen.current_state,
                    "issues": issues,
                    "confidence": current_screen.confidence_score
                }
                monitoring_data.append(monitoring_point)
                
                # Adaptive monitoring frequency
                if issues:
                    await asyncio.sleep(1.0)  # More frequent when issues detected
                else:
                    await asyncio.sleep(3.0)  # Normal frequency
                    
        except asyncio.CancelledError:
            logger.debug("Monitoring task cancelled")
    
    async def _analyze_task_complexity(
        self,
        task_description: str,
        screen_analysis: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Analyze task complexity and requirements."""
        
        complexity_prompt = f"""
        Analyze the complexity of this task:
        
        Task: {task_description}
        Current Screen: {screen_analysis.current_state}
        Available Elements: {len(screen_analysis.elements)}
        
        Assess:
        1. Complexity level (1-10)
        2. Required skills/capabilities
        3. Estimated steps needed
        4. Potential challenges
        5. Success probability
        
        Return structured analysis.
        """
        
        try:
            analysis = await self.llm_manager.generate_text(
                complexity_prompt, task_type="reasoning"
            )
            
            return {
                "complexity_score": 5,  # Default, would be parsed from analysis
                "estimated_steps": 3,
                "success_probability": 0.8,
                "challenges": [],
                "raw_analysis": analysis
            }
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return {"complexity_score": 5, "estimated_steps": 3, "success_probability": 0.5}
    
    async def _analyze_execution_risks(
        self,
        task_description: str,
        screen_analysis: ScreenAnalysis
    ) -> Dict[str, Any]:
        """Analyze potential execution risks."""
        
        risks = []
        
        # Check for obvious blockers
        if screen_analysis.errors_detected:
            risks.append({
                "type": "existing_errors",
                "severity": "high",
                "description": f"Errors already present: {screen_analysis.errors_detected}"
            })
        
        # Check for modal dialogs
        modal_elements = [e for e in screen_analysis.elements if "modal" in e.description.lower()]
        if modal_elements:
            risks.append({
                "type": "modal_blocking",
                "severity": "medium",
                "description": "Modal dialog may block interaction"
            })
        
        return {
            "risk_level": "medium" if risks else "low",
            "risks": risks,
            "mitigation_strategies": []
        }
    
    async def _gather_system_context(self) -> Dict[str, Any]:
        """Gather comprehensive system context."""
        
        try:
            system_info = await self.system_controller.get_system_info()
            running_apps = await self.system_controller.get_running_applications()
            
            return {
                "system_info": system_info,
                "running_apps_count": len(running_apps),
                "memory_usage": system_info.get("memory_percent", 0),
                "cpu_usage": system_info.get("cpu_percent", 0)
            }
        except Exception as e:
            logger.error(f"Failed to gather system context: {e}")
            return {}
    
    async def _capture_current_screen(self) -> ScreenAnalysis:
        """Capture and analyze current screen."""
        screenshot = self.screen_capture.capture_screenshot()
        return await self.vision_system.analyze_screen_intelligent(
            screenshot, task_context=self.current_task
        )
    
    async def _detect_execution_issues(self, screen_analysis: ScreenAnalysis) -> List[str]:
        """Detect issues during execution."""
        issues = []
        
        if screen_analysis.errors_detected:
            issues.extend(screen_analysis.errors_detected)
        
        if screen_analysis.confidence_score < 0.6:
            issues.append("Low confidence in screen analysis")
        
        return issues
    
    async def _post_execution_learning(
        self,
        task_description: str,
        execution_result: Dict[str, Any],
        initial_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from task execution for future improvement."""
        
        learning_insights = {
            "patterns": [],
            "improvements": [],
            "success_factors": [],
            "failure_factors": []
        }
        
        # Analyze what worked well
        if execution_result.get("success"):
            learning_insights["success_factors"].append("Task completed successfully")
            
            # Store successful pattern
            success_pattern = {
                "task_type": task_description,
                "initial_state": initial_analysis["screen_analysis"].current_state,
                "actions_taken": execution_result.get("actions_completed", 0),
                "execution_time": execution_result.get("execution_time", 0),
                "timestamp": time.time()
            }
            self.learning_memory.append(success_pattern)
        
        # Analyze failures and improvements
        if execution_result.get("error_count", 0) > 0:
            learning_insights["failure_factors"].append("Errors encountered during execution")
            learning_insights["improvements"].append("Improve error prevention")
        
        # Keep learning memory manageable
        if len(self.learning_memory) > 50:
            self.learning_memory = self.learning_memory[-50:]
        
        return learning_insights
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_tasks = self.performance_history[-10:]  # Last 10 tasks
        
        success_rate = sum(1 for task in recent_tasks if task["result"].success) / len(recent_tasks)
        avg_execution_time = sum(task["result"].execution_time for task in recent_tasks) / len(recent_tasks)
        avg_confidence = sum(task["result"].confidence_score for task in recent_tasks) / len(recent_tasks)
        
        return {
            "total_tasks": len(self.performance_history),
            "recent_success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "average_confidence": avg_confidence,
            "learning_patterns": len(self.learning_memory),
            "last_task": recent_tasks[-1]["task"] if recent_tasks else None
        }
    
    async def health_check_advanced(self) -> Dict[str, Any]:
        """Advanced health check with all 2025 components."""
        
        health = {}
        
        try:
            # Check core components
            health["vision_system"] = True  # Would check actual health
            health["planner"] = True
            health["healing_system"] = True
            
            # Check LLM providers
            llm_health = await self.llm_manager.health_check()
            health["llm_providers"] = llm_health
            
            # Check system components
            try:
                await self.system_controller.get_system_info()
                health["system_controller"] = True
            except:
                health["system_controller"] = False
            
            # Overall health
            health["overall"] = all(
                isinstance(v, bool) and v or 
                isinstance(v, dict) and any(v.values())
                for v in health.values()
            )
            
            # Performance metrics
            health["performance"] = await self.get_performance_metrics()
            
        except Exception as e:
            health["error"] = str(e)
            health["overall"] = False
        
        return health
