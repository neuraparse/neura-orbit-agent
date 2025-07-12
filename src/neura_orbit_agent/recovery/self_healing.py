"""
2025 Self-Healing and Error Recovery System
Intelligent error detection, diagnosis, and automatic recovery
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from ..vision.intelligent_vision import IntelligentVision, ScreenAnalysis
from ..core.llm_manager import LLMManager
from ..core.system_controller import SystemController
from ..utils.config import Config
from ..utils.exceptions import NeuraOrbitError
from ..utils.logger import get_logger

logger = get_logger("neura_orbit.recovery")


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    UI_ELEMENT_NOT_FOUND = "ui_element_not_found"
    NETWORK_ERROR = "network_error"
    APPLICATION_CRASH = "application_crash"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    LAYOUT_CHANGE = "layout_change"
    CAPTCHA_DETECTED = "captcha_detected"
    MODAL_BLOCKING = "modal_blocking"
    LOADING_STUCK = "loading_stuck"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = "retry"
    ALTERNATIVE_PATH = "alternative_path"
    REFRESH_PAGE = "refresh_page"
    RESTART_APPLICATION = "restart_application"
    WAIT_AND_RETRY = "wait_and_retry"
    USER_INTERVENTION = "user_intervention"
    SKIP_STEP = "skip_step"
    FALLBACK_METHOD = "fallback_method"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    screenshot_analysis: ScreenAnalysis
    timestamp: datetime
    action_attempted: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    system_state: Optional[Dict[str, Any]] = None
    previous_errors: List[str] = None


@dataclass
class RecoveryPlan:
    """Recovery plan for error resolution."""
    strategy: RecoveryStrategy
    steps: List[Dict[str, Any]]
    estimated_time: float
    success_probability: float
    fallback_plans: List['RecoveryPlan'] = None
    requires_user_input: bool = False


class SelfHealingSystem:
    """
    2025 Advanced Self-Healing and Error Recovery System.
    
    Features:
    - Intelligent error detection and classification
    - Automatic diagnosis and root cause analysis
    - Adaptive recovery strategies
    - Learning from failures
    - Predictive error prevention
    - Context-aware healing
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the self-healing system."""
        self.config = config or Config.load_default()
        
        # Core components
        self.llm_manager = LLMManager(self.config)
        self.vision_system = IntelligentVision(self.config)
        self.system_controller = SystemController(self.config)
        
        # Error tracking and learning
        self.error_history: List[ErrorContext] = []
        self.recovery_patterns: Dict[str, List[RecoveryPlan]] = {}
        self.success_rates: Dict[str, float] = {}
        
        # Adaptive thresholds
        self.error_detection_threshold = 0.8
        self.recovery_timeout = 60.0
        self.max_recovery_attempts = 3
        
        logger.info("Self-Healing System initialized with 2025 capabilities")
    
    async def detect_and_analyze_error(
        self,
        screen_analysis: ScreenAnalysis,
        action_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ErrorContext]:
        """
        Detect and analyze errors in the current screen state.
        
        Args:
            screen_analysis: Current screen analysis
            action_context: Context of the action that may have caused the error
            
        Returns:
            ErrorContext if error detected, None otherwise
        """
        try:
            # Multi-level error detection
            errors_detected = []
            
            # 1. Visual error detection
            visual_errors = await self._detect_visual_errors(screen_analysis)
            errors_detected.extend(visual_errors)
            
            # 2. Contextual error analysis
            contextual_errors = await self._analyze_contextual_errors(
                screen_analysis, action_context
            )
            errors_detected.extend(contextual_errors)
            
            # 3. Pattern-based error detection
            pattern_errors = await self._detect_pattern_errors(screen_analysis)
            errors_detected.extend(pattern_errors)
            
            if not errors_detected:
                return None
            
            # Classify and prioritize the most critical error
            primary_error = self._prioritize_errors(errors_detected)
            
            # Create comprehensive error context
            error_context = ErrorContext(
                error_id=f"error_{int(time.time())}",
                category=primary_error["category"],
                severity=primary_error["severity"],
                description=primary_error["description"],
                screenshot_analysis=screen_analysis,
                timestamp=datetime.now(),
                action_attempted=action_context.get("action") if action_context else None,
                error_message=primary_error.get("message"),
                system_state=await self._capture_system_state(),
                previous_errors=[e.error_id for e in self.error_history[-5:]]
            )
            
            # Store in history for learning
            self.error_history.append(error_context)
            if len(self.error_history) > 100:  # Keep last 100 errors
                self.error_history.pop(0)
            
            logger.warning(f"Error detected: {error_context.description}")
            return error_context
            
        except Exception as e:
            logger.error(f"Error detection failed: {e}")
            return None
    
    async def create_recovery_plan(
        self,
        error_context: ErrorContext
    ) -> Optional[RecoveryPlan]:
        """
        Create an intelligent recovery plan for the detected error.
        
        Args:
            error_context: Context of the detected error
            
        Returns:
            RecoveryPlan if recovery is possible, None otherwise
        """
        try:
            logger.info(f"Creating recovery plan for: {error_context.description}")
            
            # 1. Check for known recovery patterns
            known_recovery = self._get_known_recovery_pattern(error_context)
            if known_recovery:
                logger.info("Using known recovery pattern")
                return known_recovery
            
            # 2. AI-powered recovery planning
            ai_recovery = await self._create_ai_recovery_plan(error_context)
            if ai_recovery:
                return ai_recovery
            
            # 3. Fallback to generic recovery strategies
            generic_recovery = self._create_generic_recovery_plan(error_context)
            return generic_recovery
            
        except Exception as e:
            logger.error(f"Recovery plan creation failed: {e}")
            return None
    
    async def execute_recovery_plan(
        self,
        recovery_plan: RecoveryPlan,
        error_context: ErrorContext
    ) -> Dict[str, Any]:
        """
        Execute the recovery plan with intelligent monitoring.
        
        Args:
            recovery_plan: Recovery plan to execute
            error_context: Original error context
            
        Returns:
            Recovery execution result
        """
        try:
            logger.info(f"Executing recovery strategy: {recovery_plan.strategy.value}")
            
            start_time = time.time()
            recovery_attempts = 0
            
            while recovery_attempts < self.max_recovery_attempts:
                recovery_attempts += 1
                
                # Execute recovery steps
                step_results = []
                for i, step in enumerate(recovery_plan.steps):
                    logger.debug(f"Executing recovery step {i+1}: {step.get('description', 'Unknown')}")
                    
                    step_result = await self._execute_recovery_step(step, error_context)
                    step_results.append(step_result)
                    
                    if not step_result.get("success", False):
                        logger.warning(f"Recovery step {i+1} failed: {step_result.get('error')}")
                        break
                    
                    # Check if error is resolved after each step
                    current_screen = await self._capture_current_screen()
                    if await self._verify_error_resolved(error_context, current_screen):
                        logger.info("Error resolved successfully!")
                        return {
                            "success": True,
                            "strategy": recovery_plan.strategy.value,
                            "steps_executed": i + 1,
                            "execution_time": time.time() - start_time,
                            "attempts": recovery_attempts
                        }
                
                # If primary strategy failed, try fallback plans
                if recovery_plan.fallback_plans and recovery_attempts < self.max_recovery_attempts:
                    logger.info("Primary recovery failed, trying fallback plan")
                    for fallback_plan in recovery_plan.fallback_plans:
                        fallback_result = await self.execute_recovery_plan(fallback_plan, error_context)
                        if fallback_result.get("success"):
                            return fallback_result
                
                # Wait before retry
                if recovery_attempts < self.max_recovery_attempts:
                    await asyncio.sleep(2.0 * recovery_attempts)  # Exponential backoff
            
            # Recovery failed
            logger.error("All recovery attempts failed")
            return {
                "success": False,
                "strategy": recovery_plan.strategy.value,
                "attempts": recovery_attempts,
                "execution_time": time.time() - start_time,
                "error": "Recovery attempts exhausted"
            }
            
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    async def _detect_visual_errors(self, screen_analysis: ScreenAnalysis) -> List[Dict[str, Any]]:
        """Detect visual errors in the screen."""
        errors = []
        
        # Check for explicit error messages
        for error_msg in screen_analysis.errors_detected:
            errors.append({
                "category": ErrorCategory.VALIDATION_ERROR,
                "severity": ErrorSeverity.MEDIUM,
                "description": f"Error message detected: {error_msg}",
                "message": error_msg
            })
        
        # Check for loading indicators stuck
        if "loading" in screen_analysis.current_state.lower():
            errors.append({
                "category": ErrorCategory.LOADING_STUCK,
                "severity": ErrorSeverity.MEDIUM,
                "description": "Loading indicator detected - possible stuck state"
            })
        
        # Check for modal dialogs blocking interaction
        if any("modal" in element.description.lower() for element in screen_analysis.elements):
            errors.append({
                "category": ErrorCategory.MODAL_BLOCKING,
                "severity": ErrorSeverity.HIGH,
                "description": "Modal dialog detected - may be blocking interaction"
            })
        
        return errors
    
    async def _analyze_contextual_errors(
        self,
        screen_analysis: ScreenAnalysis,
        action_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze contextual errors based on action context."""
        errors = []
        
        if not action_context:
            return errors
        
        action_type = action_context.get("action_type")
        expected_outcome = action_context.get("expected_outcome")
        
        # Use AI to analyze if the expected outcome was achieved
        if expected_outcome:
            analysis_prompt = f"""
            Analyze if the expected outcome was achieved:
            
            Action attempted: {action_context.get('action', 'Unknown')}
            Expected outcome: {expected_outcome}
            Current screen state: {screen_analysis.current_state}
            
            Did the action succeed? If not, what went wrong?
            """
            
            try:
                analysis_result = await self.llm_manager.analyze_image(
                    screen_analysis.elements[0].bbox if screen_analysis.elements else None,
                    analysis_prompt
                )
                
                if "failed" in analysis_result.lower() or "error" in analysis_result.lower():
                    errors.append({
                        "category": ErrorCategory.UNKNOWN,
                        "severity": ErrorSeverity.MEDIUM,
                        "description": f"Action may have failed: {analysis_result[:100]}",
                        "message": analysis_result
                    })
            except Exception as e:
                logger.debug(f"Contextual analysis failed: {e}")
        
        return errors
    
    async def _detect_pattern_errors(self, screen_analysis: ScreenAnalysis) -> List[Dict[str, Any]]:
        """Detect errors based on learned patterns."""
        errors = []
        
        # Check against known error patterns from history
        for historical_error in self.error_history[-10:]:  # Check last 10 errors
            similarity = self._calculate_screen_similarity(
                screen_analysis, historical_error.screenshot_analysis
            )
            
            if similarity > 0.8:  # High similarity to previous error
                errors.append({
                    "category": historical_error.category,
                    "severity": historical_error.severity,
                    "description": f"Similar to previous error: {historical_error.description}",
                    "pattern_match": True
                })
        
        return errors
    
    def _prioritize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prioritize errors by severity and impact."""
        if not errors:
            return {}
        
        # Sort by severity (CRITICAL > HIGH > MEDIUM > LOW)
        severity_order = {
            ErrorSeverity.CRITICAL: 4,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.LOW: 1
        }
        
        sorted_errors = sorted(
            errors,
            key=lambda x: severity_order.get(x.get("severity", ErrorSeverity.LOW), 0),
            reverse=True
        )
        
        return sorted_errors[0]
    
    async def _create_ai_recovery_plan(self, error_context: ErrorContext) -> Optional[RecoveryPlan]:
        """Create AI-powered recovery plan."""
        
        recovery_prompt = f"""
        Create an intelligent recovery plan for this error:
        
        Error: {error_context.description}
        Category: {error_context.category.value}
        Severity: {error_context.severity.value}
        Screen State: {error_context.screenshot_analysis.current_state}
        Action Attempted: {error_context.action_attempted or 'Unknown'}
        
        Available Recovery Strategies:
        1. RETRY - Try the same action again
        2. ALTERNATIVE_PATH - Find alternative way to achieve goal
        3. REFRESH_PAGE - Refresh browser page
        4. RESTART_APPLICATION - Restart the application
        5. WAIT_AND_RETRY - Wait for condition then retry
        6. FALLBACK_METHOD - Use different method
        7. SKIP_STEP - Skip this step and continue
        
        Create a detailed recovery plan with:
        - Best strategy for this error type
        - Step-by-step recovery actions
        - Estimated success probability
        - Fallback options
        
        Return as structured JSON.
        """
        
        try:
            recovery_response = await self.llm_manager.generate_text(
                recovery_prompt,
                task_type="reasoning",
                max_tokens=2000
            )
            
            # Parse the response and create RecoveryPlan
            plan_data = self._parse_recovery_response(recovery_response)
            
            if plan_data:
                return RecoveryPlan(
                    strategy=RecoveryStrategy(plan_data.get("strategy", "retry")),
                    steps=plan_data.get("steps", []),
                    estimated_time=plan_data.get("estimated_time", 30.0),
                    success_probability=plan_data.get("success_probability", 0.5),
                    requires_user_input=plan_data.get("requires_user_input", False)
                )
            
        except Exception as e:
            logger.error(f"AI recovery plan creation failed: {e}")
        
        return None
    
    def _parse_recovery_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI recovery response."""
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        return {
            "strategy": "retry",
            "steps": [{"action": "retry", "description": "Retry the failed action"}],
            "estimated_time": 10.0,
            "success_probability": 0.6
        }
    
    async def _capture_current_screen(self) -> ScreenAnalysis:
        """Capture and analyze current screen."""
        from ..core.screen_capture import ScreenCapture
        
        screen_capture = ScreenCapture(self.config)
        screenshot = screen_capture.capture_screenshot()
        
        return await self.vision_system.analyze_screen_intelligent(screenshot)
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context."""
        try:
            system_info = await self.system_controller.get_system_info()
            running_apps = await self.system_controller.get_running_applications()
            
            return {
                "system_info": system_info,
                "running_applications": len(running_apps),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {}
    
    def _calculate_screen_similarity(
        self,
        screen1: ScreenAnalysis,
        screen2: ScreenAnalysis
    ) -> float:
        """Calculate similarity between two screen analyses."""
        # Simple similarity based on current state and error messages
        state_similarity = 1.0 if screen1.current_state == screen2.current_state else 0.0
        error_similarity = len(set(screen1.errors_detected) & set(screen2.errors_detected)) / max(
            len(screen1.errors_detected) + len(screen2.errors_detected), 1
        )
        
        return (state_similarity + error_similarity) / 2.0
