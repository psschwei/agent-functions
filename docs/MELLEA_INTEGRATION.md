# Mellea Integration Guide

This document describes the integration of [Mellea](https://github.com/generative-computing/mellea) into the agent-functions project for adaptive execution and result evaluation.

## Overview

Mellea is IBM Research's generative programming library that enables LLM-based code generation with structured constraints. In this project, Mellea enhances the `ClassicalAgent` with:

- **Adaptive Execution**: Automatic retry logic with LLM-guided parameter adjustments
- **Result Quality Evaluation**: LLM-based assessment of stage outputs
- **Parameter Optimization**: Intelligent suggestions for improving results

## Architecture

### MelleaClassicalAgent

The `MelleaClassicalAgent` extends `ClassicalAgent` with Mellea capabilities:

```
ClassicalAgent (base)
    ↓
MelleaClassicalAgent (enhanced)
    ↓
    ├─ Result Evaluation (LLM-based quality assessment)
    ├─ Adjustment Suggestions (parameter optimization)
    └─ Adaptive Retry Logic (up to N attempts with improvements)
```

### Workflow Integration

The workflow automatically detects Mellea configuration and uses the enhanced agent when enabled:

```python
# In pattern_graph.py map_stage_node
if MELLEA_CONFIG["enabled"] and "map" in MELLEA_CONFIG["stages"]:
    agent = MelleaClassicalAgent(...)  # Enhanced agent
else:
    agent = ClassicalAgent(...)  # Standard agent
```

## Configuration

### Enable Mellea

Edit `config/settings.py`:

```python
MELLEA_CONFIG = {
    "enabled": True,  # Enable Mellea agents
    "model_backend": "ollama",  # Backend: ollama, watsonx, huggingface, openai
    "max_retries": 2,  # Maximum adaptive retries per stage
    "stages": ["map"],  # Which stages to use Mellea for
    "model_name": "llama2",  # Model name for the backend
    "temperature": 0.7,  # Temperature for LLM generation
    "evaluation_enabled": True,  # Enable result quality evaluation
    "adjustment_enabled": True,  # Enable parameter adjustment suggestions
}
```

### Supported Backends

Mellea supports multiple LLM backends:

- **ollama**: Local Ollama server (requires Ollama running)
- **watsonx**: IBM watsonx.ai
- **huggingface**: HuggingFace models
- **openai**: OpenAI API

### Stage Selection

You can enable Mellea for specific stages:

```python
"stages": ["map"]  # Only map stage
"stages": ["map", "optimize", "post_process"]  # Multiple classical stages
```

**Note**: Mellea is designed for classical stages. The `execute` stage (quantum) should use the standard `QuantumAgent`.

## Usage

### Basic Usage

1. **Install Mellea** (already in dependencies):
   ```bash
   uv sync
   ```

2. **Start Ollama** (if using ollama backend):
   ```bash
   ollama serve
   ollama pull llama2
   ```

3. **Enable Mellea in config**:
   ```python
   MELLEA_CONFIG["enabled"] = True
   ```

4. **Run the pattern**:
   ```bash
   python main.py --pattern chsh
   ```

### Example Output

With Mellea enabled, you'll see enhanced logging:

```
============================================================
MAP STAGE
============================================================
[Workflow] Using Mellea-enhanced agent for map stage
[MelleaClassicalAgent-Map] Initialized Mellea session with ollama backend
[MelleaClassicalAgent-Map] Executing decorated map stage with Mellea adaptation...
[MelleaClassicalAgent-Map] Attempt 1/3
[MelleaClassicalAgent-Map] Quality evaluation: satisfactory
[MelleaClassicalAgent-Map] ✓ map stage completed in 2.34s
```

If quality is insufficient:

```
[MelleaClassicalAgent-Map] Quality evaluation: needs_improvement
[MelleaClassicalAgent-Map] Result quality insufficient, suggesting adjustments...
[MelleaClassicalAgent-Map] Suggested adjustments: {'parameter_sweep_points': 50}
[MelleaClassicalAgent-Map] Applied adjustment: parameter_sweep_points = 50
[MelleaClassicalAgent-Map] Attempt 2/3
```

## How It Works

### 1. Result Evaluation

After each stage execution, Mellea evaluates the result quality:

```python
evaluation = {
    "quality": "satisfactory" | "needs_improvement",
    "issues": ["list", "of", "identified", "issues"],
    "reasoning": "Brief explanation of assessment"
}
```

The evaluation considers:
- Data completeness (all expected keys present)
- Value validity (no None/NaN where unexpected)
- Logical consistency (values make sense for the stage)

### 2. Adjustment Suggestions

If quality is insufficient, Mellea suggests parameter adjustments:

```python
adjustments = {
    "adjustments": {
        "parameter_name": new_value,
        ...
    },
    "reasoning": "Why these adjustments should help"
}
```

### 3. Adaptive Retry

The agent applies adjustments and retries up to `max_retries` times:

```python
for attempt in range(max_retries + 1):
    result = execute_stage()
    evaluation = evaluate_quality(result)
    
    if evaluation["quality"] == "satisfactory":
        return result
    
    if attempt < max_retries:
        adjustments = suggest_adjustments(evaluation)
        apply_adjustments(adjustments)
        continue  # Retry
```

## Comparison with Standard Agent

| Feature | ClassicalAgent | MelleaClassicalAgent |
|---------|---------------|---------------------|
| Execution | Single attempt | Adaptive retry (up to N attempts) |
| Quality Check | None | LLM-based evaluation |
| Parameter Tuning | Manual | Automatic suggestions |
| Error Recovery | Fail immediately | Retry with adjustments |
| Overhead | Minimal | LLM inference time |

## Performance Considerations

### Overhead

Mellea adds overhead from LLM inference:
- **Evaluation**: ~1-3 seconds per evaluation
- **Adjustment**: ~1-3 seconds per suggestion
- **Total**: ~2-6 seconds per retry attempt

### When to Use Mellea

**Good Use Cases:**
- Exploratory workflows where optimal parameters are unknown
- Patterns with sensitive parameter dependencies
- Development/debugging to understand failure modes

**Not Recommended:**
- Production workflows with known-good parameters
- Time-critical applications
- Stages with deterministic, well-tested logic

## Troubleshooting

### Mellea Not Available

If you see:
```
Warning: Mellea not installed. MelleaClassicalAgent will not be available.
```

**Solution**: Mellea is installed but the import failed. Check:
1. Virtual environment is activated
2. Dependencies are installed: `uv sync`

### Ollama Connection Failed

If you see:
```
Warning: Failed to initialize Mellea session: Connection refused
```

**Solution**: Start Ollama server:
```bash
ollama serve
```

### Fallback Behavior

If Mellea initialization fails, the agent automatically falls back to standard `ClassicalAgent` behavior:

```python
if self.session is None:
    # Fallback to standard execution
    return super().run_decorated_stage(stage_func, ctx, stage_name)
```

## Testing

### Integration Test

Run the integration test to verify Mellea setup:

```bash
python test_mellea_integration.py
```

Expected output:
```
✓ PASS: Mellea Import
✓ PASS: MelleaClassicalAgent Import
✓ PASS: MelleaClassicalAgent Instantiation
✓ PASS: Configuration Loading
✓ PASS: Workflow Integration

Total: 5/5 tests passed
✓ All tests passed! Mellea integration is ready.
```

### Comparison Script

Compare Mellea vs standard execution:

```bash
python scripts/compare_mellea_execution.py --pattern chsh --runs 3
```

This will run the pattern multiple times with and without Mellea, comparing:
- Execution time
- Result quality
- Number of retries
- Success rate

## API Reference

### MelleaClassicalAgent

```python
class MelleaClassicalAgent(ClassicalAgent):
    def __init__(
        self,
        name: str = "MelleaClassicalAgent",
        model_backend: str = "ollama",
        max_retries: int = 2
    )
```

**Parameters:**
- `name`: Agent name for logging
- `model_backend`: Mellea backend (ollama, watsonx, huggingface, openai)
- `max_retries`: Maximum adaptive retries per stage

**Methods:**
- `run_decorated_stage()`: Execute stage with adaptive retry
- `_evaluate_result_quality()`: LLM-based quality evaluation
- `_suggest_adjustments()`: LLM-based parameter suggestions
- `_apply_adjustments()`: Apply adjustments to context

### Configuration

```python
MELLEA_CONFIG = {
    "enabled": bool,  # Enable/disable Mellea
    "model_backend": str,  # Backend name
    "max_retries": int,  # Max retries per stage
    "stages": list[str],  # Stages to enhance
    "model_name": str,  # Model name
    "temperature": float,  # LLM temperature
    "evaluation_enabled": bool,  # Enable evaluation
    "adjustment_enabled": bool,  # Enable adjustments
}
```

## Future Enhancements

Potential improvements for Mellea integration:

1. **Multi-Stage Coordination**: Share learnings across stages
2. **Parameter History**: Track successful parameter combinations
3. **Custom Evaluation Metrics**: Pattern-specific quality criteria
4. **Parallel Evaluation**: Evaluate multiple parameter sets simultaneously
5. **Cost Tracking**: Monitor LLM API costs

## References

- [Mellea GitHub Repository](https://github.com/generative-computing/mellea)
- [Mellea Documentation](https://github.com/generative-computing/mellea#readme)
- [Issue #6: Use Mellea agents](https://github.com/psschwei/agent-functions/issues/6)

## Contributing

To extend Mellea integration:

1. Add new evaluation criteria in `_evaluate_result_quality()`
2. Enhance adjustment logic in `_suggest_adjustments()`
3. Add pattern-specific evaluation prompts
4. Implement custom Mellea strategies

See `agents/mellea_classical_agent.py` for implementation details.
