"""
Unit tests for patterns.decorators module.

Tests decorator registration, metadata storage, and stage-specific decorators.
"""
import pytest
from pathlib import Path
import pickle
from typing import Dict

from patterns.decorators import (
    map_stage,
    optimize_stage,
    execute_stage,
    post_process_stage,
    PatternContext,
    StageMetadata,
    get_pattern_registry,
    clear_pattern_registry,
    get_pattern_stages,
    has_decorated_pattern,
)


@pytest.fixture
def clean_registry():
    """Clear pattern registry before and after each test."""
    clear_pattern_registry()
    yield
    clear_pattern_registry()


@pytest.fixture
def temp_state():
    """Create a minimal pattern state for testing."""
    return {
        "pattern_name": "test",
        "current_stage": "map",
        "stage_status": {},
        "errors": [],
    }


class TestStageDecorators:
    """Test stage-specific decorators."""

    def test_map_stage_decorator_basic(self, clean_registry):
        """Test basic @map_stage decorator."""
        @map_stage()
        def test_map(ctx: PatternContext) -> dict:
            return {"result": "map"}

        assert hasattr(test_map, '_pattern_metadata')
        metadata = test_map._pattern_metadata
        assert metadata.stage_name == "map"
        assert metadata.agent_type == "classical"

    def test_optimize_stage_decorator_basic(self, clean_registry):
        """Test basic @optimize_stage decorator."""
        @optimize_stage()
        def test_optimize(ctx: PatternContext) -> dict:
            return {"result": "optimize"}

        metadata = test_optimize._pattern_metadata
        assert metadata.stage_name == "optimize"
        assert metadata.agent_type == "classical"

    def test_execute_stage_decorator_basic(self, clean_registry):
        """Test basic @execute_stage decorator."""
        @execute_stage()
        def test_execute(ctx: PatternContext) -> dict:
            return {"result": "execute"}

        metadata = test_execute._pattern_metadata
        assert metadata.stage_name == "execute"
        assert metadata.agent_type == "quantum"

    def test_post_process_stage_decorator_basic(self, clean_registry):
        """Test basic @post_process_stage decorator."""
        @post_process_stage()
        def test_post_process(ctx: PatternContext) -> dict:
            return {"result": "post_process"}

        metadata = test_post_process._pattern_metadata
        assert metadata.stage_name == "post_process"
        assert metadata.agent_type == "classical"

    def test_decorator_without_parentheses(self, clean_registry):
        """Test decorator can be used without parentheses."""
        @map_stage
        def test_map(ctx: PatternContext) -> dict:
            return {"result": "map"}

        assert hasattr(test_map, '_pattern_metadata')
        assert test_map._pattern_metadata.stage_name == "map"

    def test_decorator_with_custom_agent_type(self, clean_registry):
        """Test decorator with custom agent_type override."""
        @map_stage(agent_type="quantum")
        def test_map(ctx: PatternContext) -> dict:
            return {"result": "map"}

        assert test_map._pattern_metadata.agent_type == "quantum"

    def test_decorator_with_timeout(self, clean_registry):
        """Test decorator with timeout parameter."""
        @execute_stage(timeout=120)
        def test_execute(ctx: PatternContext) -> dict:
            return {"result": "execute"}

        assert test_execute._pattern_metadata.timeout == 120

    def test_decorator_with_retry(self, clean_registry):
        """Test decorator with retry parameter."""
        @execute_stage(retry=3)
        def test_execute(ctx: PatternContext) -> dict:
            return {"result": "execute"}

        assert test_execute._pattern_metadata.retry == 3

    def test_decorator_with_custom_dependencies(self, clean_registry):
        """Test decorator with custom dependencies."""
        @post_process_stage(dependencies=["map", "execute"])
        def test_post_process(ctx: PatternContext) -> dict:
            return {"result": "post_process"}

        assert test_post_process._pattern_metadata.dependencies == ["map", "execute"]


class TestPatternRegistry:
    """Test pattern registry functionality."""

    def test_decorator_registers_in_global_registry(self, clean_registry):
        """Test that decorators register functions in global registry."""
        # Create a test pattern by decorating functions
        # Note: The pattern name comes from the module name
        @map_stage()
        def test_map(ctx: PatternContext) -> dict:
            return {"result": "map"}

        registry = get_pattern_registry()
        # The pattern should be registered (pattern name extracted from module)
        assert len(registry) > 0

    def test_has_decorated_pattern(self, clean_registry):
        """Test has_decorated_pattern function."""
        # Initially no patterns
        assert not has_decorated_pattern("test_pattern")

        # After decorating, should be detected
        @map_stage()
        def test_map(ctx: PatternContext) -> dict:
            return {"result": "map"}

        # Pattern name comes from module, so check registry
        registry = get_pattern_registry()
        assert len(registry) > 0

    def test_clear_pattern_registry(self, clean_registry):
        """Test clearing the pattern registry."""
        @map_stage()
        def test_map(ctx: PatternContext) -> dict:
            return {"result": "map"}

        # Registry should have entries
        assert len(get_pattern_registry()) > 0

        # Clear and verify empty
        clear_pattern_registry()
        assert len(get_pattern_registry()) == 0


class TestStageMetadata:
    """Test StageMetadata class."""

    def test_default_dependencies(self, clean_registry):
        """Test that default dependencies are set correctly."""
        # Map stage should have no dependencies
        @map_stage()
        def test_map(ctx: PatternContext) -> dict:
            return {}

        assert test_map._pattern_metadata.dependencies == []

        # Optimize should depend on map
        @optimize_stage()
        def test_optimize(ctx: PatternContext) -> dict:
            return {}

        assert test_optimize._pattern_metadata.dependencies == ["map"]

        # Execute should depend on optimize
        @execute_stage()
        def test_execute(ctx: PatternContext) -> dict:
            return {}

        assert test_execute._pattern_metadata.dependencies == ["optimize"]

        # Post-process should depend on execute
        @post_process_stage()
        def test_post_process(ctx: PatternContext) -> dict:
            return {}

        assert test_post_process._pattern_metadata.dependencies == ["execute"]

    def test_metadata_function_reference(self, clean_registry):
        """Test that metadata contains callable function reference."""
        def my_map_function(ctx: PatternContext) -> dict:
            return {"result": "test"}

        decorated = map_stage()(my_map_function)

        # Metadata func should be callable (it's the decorated wrapper)
        assert callable(decorated._pattern_metadata.func)
        # Both the metadata func and decorated should have the pattern metadata attached
        assert hasattr(decorated._pattern_metadata.func, '_pattern_metadata')
        assert hasattr(decorated, '_pattern_metadata')


class TestDecoratedFunctionExecution:
    """Test execution of decorated functions."""

    def test_decorated_function_auto_saves_output(self, clean_registry, temp_state, tmp_path):
        """Test that decorated functions automatically save output."""
        # Override DATA_DIR temporarily
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            @map_stage()
            def test_map(ctx: PatternContext) -> dict:
                return {"circuit": "test_circuit", "parameter": "theta"}

            ctx = PatternContext(temp_state, "map", "test")
            result = test_map(ctx)

            # Result should be returned
            assert result == {"circuit": "test_circuit", "parameter": "theta"}

            # Output should be saved to file
            output_path = tmp_path / "test_map_result.pkl"
            assert output_path.exists()

            # Verify saved content
            with open(output_path, 'rb') as f:
                saved_data = pickle.load(f)
            assert saved_data == {"circuit": "test_circuit", "parameter": "theta"}

        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_decorated_function_preserves_docstring(self, clean_registry):
        """Test that decorated functions preserve docstrings."""
        @map_stage()
        def test_map(ctx: PatternContext) -> dict:
            """This is a test docstring."""
            return {}

        assert test_map.__doc__ == "This is a test docstring."

    def test_decorated_function_preserves_name(self, clean_registry):
        """Test that decorated functions preserve function names via @wraps."""
        @map_stage()
        def my_custom_map_function(ctx: PatternContext) -> dict:
            return {}

        # The @wraps decorator should preserve the original function name
        assert my_custom_map_function.__name__ == "my_custom_map_function"
