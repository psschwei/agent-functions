"""
Agentic Orchestrator with LLM-based decision making.

Extends the basic Orchestrator with autonomous reasoning using LLMs.
"""
from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path

from agents.orchestrator import Orchestrator
from agents.llm_client import LLMClient
from agents.tools import TOOL_FUNCTIONS, TOOL_DEFINITIONS
from config import LLM_CONFIG, ORCHESTRATOR_CONFIG, DATA_DIR
from workflows.pattern_graph import PatternState, create_pattern_workflow, create_initial_state

ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert quantum computing orchestrator coordinating the execution of Qiskit patterns.

Your role: Make strategic decisions about workflow execution, parameter selection, and optimization strategies for quantum circuit experiments.

Current pattern: CHSH inequality test (Bell state entanglement verification)

Your responsibilities:
1. Analyze circuit complexity to recommend optimization strategies
2. Evaluate intermediate stage results for quality and completeness
3. Recommend optimal parameters for upcoming stages
4. Decide whether to retry failed or suboptimal stages
5. Determine when to terminate vs. iterate the workflow

Key principles:
- Prioritize scientific accuracy over execution speed
- Make conservative decisions when uncertain
- Always explain your reasoning
- Consider resource constraints (shots, execution time)
- Balance exploration (more measurements) vs. exploitation (quick results)

CHSH-specific knowledge:
- Classical bound: 2.0
- Quantum bound: 2√2 ≈ 2.828
- Typical violation: 2.5-2.8 with finite shots
- Simple Bell state circuits: 4-6 gates (H, CNOT, RY, measure)
- Standard observables: ZZ and XX (Pauli measurements)
- Phase sampling: 8-64 points typical (16 is standard)

Make decisions autonomously without asking for user confirmation.
"""


class AgenticOrchestrator(Orchestrator):
    """
    Agentic Orchestrator with LLM-powered reasoning.

    Extends basic Orchestrator with autonomous decision-making capabilities.
    """

    def __init__(
        self,
        pattern_name: str = "chsh",
        enable_llm: bool = None,
        llm_config: Dict[str, Any] = None,
    ):
        """
        Initialize agentic orchestrator.

        Args:
            pattern_name: Name of the pattern to execute
            enable_llm: Enable LLM reasoning (overrides config)
            llm_config: LLM configuration (overrides defaults)
        """
        super().__init__(pattern_name)

        # LLM configuration
        config = llm_config or LLM_CONFIG
        self.enable_llm = enable_llm if enable_llm is not None else ORCHESTRATOR_CONFIG["enable_llm"]

        # Initialize LLM client
        if self.enable_llm:
            self.llm_client = LLMClient(
                model=config["model"],
                base_url=config.get("base_url"),
                api_key=config.get("api_key"),
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config.get("timeout", 30),
                max_retries=config.get("max_retries", 3),
            )
        else:
            self.llm_client = None

        # Decision logging
        self.decision_log = []
        self.decision_log_path = ORCHESTRATOR_CONFIG.get("decision_log_path")
        self.total_tokens = 0

    def initialize_workflow(self):
        """Initialize the LangGraph workflow with LLM-enabled state."""
        print("\n" + "=" * 60)
        print("AGENTIC ORCHESTRATOR: Initializing Workflow")
        print("=" * 60)
        print(f"Pattern: {self.pattern_name}")
        print(f"LLM Enabled: {self.enable_llm}")

        # Create workflow
        self.workflow = create_pattern_workflow()

        # Create initial state with LLM enabled
        self.state = create_initial_state(self.pattern_name, enable_llm=self.enable_llm)

        print("✓ Workflow initialized")
        print(f"  Stages: {list(self.state['stage_status'].keys())}")
        if self.enable_llm:
            print(f"  LLM Model: {self.llm_client.model if self.llm_client else 'N/A'}")

    def reason_before_stage(
        self,
        stage_name: str,
        state: PatternState,
    ) -> Dict[str, Any]:
        """
        LLM reasoning before executing a stage.

        Asks: "What parameters should I use for this stage?"

        Args:
            stage_name: Name of upcoming stage
            state: Current workflow state

        Returns:
            Dictionary with recommended parameters and reasoning
        """
        if not self.enable_llm:
            return self._fallback_before_stage(stage_name, state)

        print(f"\n[Orchestrator] 🤔 Reasoning about {stage_name} stage parameters...")

        # Build context from state
        context = self._build_context(state)

        # Construct prompt
        user_prompt = f"""I'm about to execute the {stage_name} stage of the CHSH workflow.

Current state:
- Pattern: {state['pattern_name']}
- Completed stages: {[s for s, st in state['stage_status'].items() if st == 'complete']}
- Previous timings: {state.get('stage_timings', {})}

Please recommend optimal parameters for the {stage_name} stage.

Use the recommend_parameters tool to generate your recommendation."""

        # Run ReAct loop
        try:
            result = self._react_loop(
                user_prompt=user_prompt,
                context=context,
                max_iterations=5,
            )

            # Log decision
            self._log_decision(
                decision_type="before_stage",
                stage_name=stage_name,
                reasoning=result.get("reasoning", ""),
                parameters=result.get("parameters", {}),
            )

            print(f"[Orchestrator] 💭 Decision: {result.get('reasoning', 'No reasoning provided')}")

            return result

        except Exception as e:
            print(f"[Orchestrator] ⚠️  LLM reasoning failed: {str(e)}")
            print(f"[Orchestrator] ⚠️  Falling back to defaults")
            return self._fallback_before_stage(stage_name, state)

    def reason_after_stage(
        self,
        stage_name: str,
        state: PatternState,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        LLM reasoning after stage execution.

        Asks: "Did this work well? Should I retry or proceed?"

        Args:
            stage_name: Name of completed stage
            state: Current workflow state
            result: Stage execution result

        Returns:
            Dictionary with evaluation and retry decision
        """
        if not self.enable_llm:
            return self._fallback_after_stage(stage_name, state, result)

        print(f"\n[Orchestrator] 🔍 Evaluating {stage_name} stage results...")

        # Build context
        context = self._build_context(state)
        context["stage_result"] = result

        # Construct prompt
        output_path = state.get(f"{stage_name}_output")
        user_prompt = f"""The {stage_name} stage just completed.

Result summary:
- Status: {result.get('status')}
- Duration: {result.get('duration', 0):.2f}s
- Output: {output_path}

Please evaluate the stage results using the evaluate_stage_results tool.
Then decide: should we proceed to the next stage, or retry this stage with different parameters?"""

        # Run ReAct loop
        try:
            evaluation = self._react_loop(
                user_prompt=user_prompt,
                context=context,
                max_iterations=5,
            )

            # Log decision
            self._log_decision(
                decision_type="after_stage",
                stage_name=stage_name,
                evaluation=evaluation,
            )

            if evaluation.get("should_retry"):
                print(f"[Orchestrator] 🔄 Decision: Retry recommended")
                print(f"[Orchestrator] 💭 Reason: {evaluation.get('reasoning', 'Unknown')}")
            else:
                print(f"[Orchestrator] ✓ Decision: Proceed to next stage")

            return evaluation

        except Exception as e:
            print(f"[Orchestrator] ⚠️  LLM evaluation failed: {str(e)}")
            return self._fallback_after_stage(stage_name, state, result)

    def _react_loop(
        self,
        user_prompt: str,
        context: Dict[str, Any],
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute ReAct loop (Reason + Act).

        Args:
            user_prompt: User's question/task
            context: Context dictionary
            max_iterations: Maximum tool calls to prevent infinite loops

        Returns:
            Final result from LLM
        """
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(max_iterations):
            # Call LLM with tools
            response = self.llm_client.chat_completion(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )

            # Track token usage
            self._track_usage(response["usage"])

            # Check if LLM wants to use tools
            if response["tool_calls"]:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response["tool_calls"]
                    ]
                })

                # Execute tool calls
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"[Orchestrator] 🔧 Calling tool: {tool_name}")

                    # Execute tool
                    tool_func = TOOL_FUNCTIONS.get(tool_name)
                    if tool_func:
                        tool_result = tool_func(**tool_args)
                    else:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    })
            else:
                # LLM provided final answer
                final_content = response["content"]
                if final_content:
                    print(f"[Orchestrator] 💡 LLM response: {final_content}")

                    # Parse response (expect JSON or extract key information)
                    try:
                        return json.loads(final_content)
                    except json.JSONDecodeError:
                        # Return as structured dict
                        return {
                            "reasoning": final_content,
                            "parameters": {},
                        }
                else:
                    # No content and no tool calls - unexpected
                    return {
                        "reasoning": "LLM provided no response",
                        "parameters": {},
                    }

        # Max iterations reached
        print(f"[Orchestrator] ⚠️  Max iterations ({max_iterations}) reached")
        return {
            "reasoning": "Max iterations reached",
            "parameters": {},
        }

    def _build_context(self, state: PatternState) -> Dict[str, Any]:
        """Build context dictionary from state."""
        return {
            "pattern_name": state["pattern_name"],
            "current_stage": state["current_stage"],
            "stage_status": state["stage_status"],
            "stage_timings": state["stage_timings"],
            "errors": state["errors"],
        }

    def _fallback_before_stage(
        self,
        stage_name: str,
        state: PatternState,
    ) -> Dict[str, Any]:
        """Fallback parameters when LLM is disabled or fails."""
        # Use hardcoded defaults per stage
        defaults = {
            "map": {"phase_count": 16, "observables": ["ZZ", "XX"]},
            "optimize": {"optimization_level": 1},
            "execute": {"shots": 1024, "execution_strategy": "batched"},
            "post_process": {"plot_types": ["standard"], "dpi": 150},
        }

        return {
            "parameters": defaults.get(stage_name, {}),
            "reasoning": "Using default parameters (LLM disabled or failed)",
        }

    def _fallback_after_stage(
        self,
        stage_name: str,
        state: PatternState,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback evaluation when LLM is disabled or fails."""
        # Simple rule: proceed if status is success, otherwise retry
        should_retry = result.get("status") != "success"

        return {
            "should_retry": should_retry,
            "quality_score": 1.0 if not should_retry else 0.0,
            "reasoning": "Using rule-based evaluation (LLM disabled or failed)",
        }

    def _log_decision(self, **kwargs):
        """Log LLM decision for audit trail."""
        decision = {
            "timestamp": time.time(),
            **kwargs,
        }
        self.decision_log.append(decision)

        # Write to file if path configured
        if ORCHESTRATOR_CONFIG.get("enable_logging") and self.decision_log_path:
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.decision_log_path, 'a') as f:
                f.write(json.dumps(decision) + '\n')

    def _track_usage(self, usage: Dict[str, int]):
        """Track token usage for cost monitoring."""
        self.total_tokens += usage["total_tokens"]
        print(f"[Orchestrator] 📊 Tokens: {usage['total_tokens']} (total: {self.total_tokens})")

    def reflect_on_workflow(self, state: PatternState) -> Dict[str, Any]:
        """
        Reflect on the complete workflow and decide whether to iterate.

        Phase 3: Analyzes the entire workflow execution and determines if
        another iteration with different parameters would improve results.

        Args:
            state: Complete workflow state after all stages

        Returns:
            Dictionary with reflection analysis and iteration decision:
            - should_iterate: bool - whether to run another iteration
            - reasoning: str - explanation of the decision
            - recommended_changes: dict - suggested parameter changes
        """
        if not self.enable_llm:
            return self._fallback_reflection(state)

        print(f"\n[Orchestrator] 🤔 Reflecting on complete workflow...")

        # Extract key results from state
        iteration = state.get("workflow_iteration", 1)
        timings = state.get("stage_timings", {})
        errors = state.get("errors", [])
        total_duration = sum(timings.values())

        # Try to extract CHSH-specific results (pattern-specific)
        chsh_violation = None
        max_chsh_value = None
        try:
            import pickle
            from pathlib import Path
            post_process_output = state.get("post_process_output")
            if post_process_output:
                # Try to load summary
                data_path = Path(post_process_output).parent / f"{state['pattern_name']}_post_process_result.pkl"
                if data_path.exists():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                        if "summary" in data:
                            max_chsh_value = data["summary"].get("max_chsh_value")
                            chsh_violation = data["summary"].get("violation_detected")
        except Exception as e:
            print(f"[Orchestrator] ⚠️  Could not load results: {e}")

        # Build reflection prompt
        user_prompt = f"""The complete workflow for pattern '{state['pattern_name']}' has finished.

Workflow Summary (Iteration {iteration}):
- Total execution time: {total_duration:.2f}s
- Stage timings: {timings}
- Errors encountered: {len(errors)}
- CHSH violation detected: {chsh_violation}
- Maximum CHSH value: {max_chsh_value}

Previous iterations: {len(state.get('iteration_history', []))}

Questions:
1. Should I run another iteration with different parameters?
2. What parameter changes would improve results?
3. Is the current result satisfactory, or can we do better?

For CHSH pattern:
- Target: Maximize CHSH violation (ideally ~2.828)
- Classical bound: 2.0
- Quantum bound: 2.828

Provide your reflection as JSON with fields:
- should_iterate: true/false
- reasoning: explanation of decision
- recommended_changes: {{stage: {{param: new_value}}}} (empty if not iterating)
- confidence: low/medium/high
"""

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm_client.chat_completion(messages, tools=None)
            self._track_usage(response["usage"])

            content = response["content"]
            if content:
                # Try to parse JSON from response
                import json
                import re
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    reflection = json.loads(json_match.group())
                    print(f"[Orchestrator] 💭 Reflection: {reflection.get('reasoning', 'N/A')}")
                    print(f"[Orchestrator] 🎯 Confidence: {reflection.get('confidence', 'N/A')}")

                    self._log_decision(
                        decision_type="workflow_reflection",
                        stage_name="complete_workflow",
                        decision=f"Iterate: {reflection.get('should_iterate', False)}",
                        reasoning=reflection.get("reasoning", ""),
                        parameters=reflection.get("recommended_changes", {}),
                        usage=response["usage"],
                    )

                    return reflection

            # If parsing failed, don't iterate
            return {
                "should_iterate": False,
                "reasoning": "Could not parse LLM response",
                "recommended_changes": {},
                "confidence": "low"
            }

        except Exception as e:
            print(f"[Orchestrator] ❌ Reflection failed: {e}")
            return self._fallback_reflection(state)

    def _fallback_reflection(self, state: PatternState) -> Dict[str, Any]:
        """
        Fallback reflection when LLM is disabled or fails.

        Simple heuristic: don't iterate unless results are clearly poor.
        """
        print(f"[Orchestrator] 💭 Using fallback reflection (LLM disabled)")

        # Simple heuristic: check if there were errors
        errors = state.get("errors", [])
        has_errors = len(errors) > 0

        return {
            "should_iterate": has_errors,
            "reasoning": "Errors detected - retry recommended" if has_errors else "No issues detected - workflow complete",
            "recommended_changes": {},
            "confidence": "low"
        }
