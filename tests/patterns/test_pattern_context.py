"""
Unit tests for PatternContext class.

Tests I/O operations, path management, logging, and config access.
"""
import pytest
import pickle
from pathlib import Path

from patterns.decorators import PatternContext


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_state():
    """Create a mock pattern state."""
    return {
        "pattern_name": "test_pattern",
        "current_stage": "optimize",
        "map_output": "data/test_pattern_map_result.pkl",
        "stage_status": {"map": "complete", "optimize": "running"},
        "errors": [],
        "custom_config": "test_value",
    }


class TestPatternContextInit:
    """Test PatternContext initialization."""

    def test_init_basic(self, mock_state):
        """Test basic initialization."""
        ctx = PatternContext(mock_state, "optimize", "test_pattern")

        assert ctx.state == mock_state
        assert ctx.stage_name == "optimize"
        assert ctx.pattern_name == "test_pattern"
        assert ctx._logs == []

    def test_init_creates_log_list(self, mock_state):
        """Test that initialization creates empty log list."""
        ctx = PatternContext(mock_state, "map", "test_pattern")
        assert isinstance(ctx._logs, list)
        assert len(ctx._logs) == 0


class TestPatternContextPaths:
    """Test path management methods."""

    def test_get_output_path(self, mock_state, tmp_path):
        """Test get_output_path returns correct path."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "optimize", "test_pattern")
            output_path = ctx.get_output_path()

            assert output_path == tmp_path / "test_pattern_optimize_result.pkl"
            assert isinstance(output_path, Path)
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_get_input_path_for_first_stage(self, mock_state, tmp_path):
        """Test get_input_path returns None for map stage."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "map", "test_pattern")
            input_path = ctx.get_input_path()

            assert input_path is None
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_get_input_path_for_second_stage(self, mock_state, tmp_path):
        """Test get_input_path returns correct path for optimize stage."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "optimize", "test_pattern")
            input_path = ctx.get_input_path()

            assert input_path == tmp_path / "test_pattern_map_result.pkl"
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_get_input_path_for_execute_stage(self, mock_state, tmp_path):
        """Test get_input_path returns correct path for execute stage."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "execute", "test_pattern")
            input_path = ctx.get_input_path()

            assert input_path == tmp_path / "test_pattern_optimize_result.pkl"
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_get_input_path_for_post_process_stage(self, mock_state, tmp_path):
        """Test get_input_path returns correct path for post_process stage."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "post_process", "test_pattern")
            input_path = ctx.get_input_path()

            assert input_path == tmp_path / "test_pattern_execute_result.pkl"
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_get_input_path_raises_for_unknown_stage(self, mock_state):
        """Test get_input_path raises ValueError for unknown stage."""
        ctx = PatternContext(mock_state, "invalid_stage", "test_pattern")

        with pytest.raises(ValueError, match="Unknown stage"):
            ctx.get_input_path()


class TestPatternContextIO:
    """Test I/O operations."""

    def test_save_output(self, mock_state, tmp_path):
        """Test save_output writes pickle file."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "map", "test_pattern")
            test_data = {"circuit": "test", "params": [1, 2, 3]}

            output_path = ctx.save_output(test_data)

            # Check file was created
            assert output_path.exists()
            assert output_path == tmp_path / "test_pattern_map_result.pkl"

            # Verify content
            with open(output_path, 'rb') as f:
                loaded_data = pickle.load(f)
            assert loaded_data == test_data
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_save_output_creates_parent_directory(self, mock_state, tmp_path):
        """Test save_output creates parent directory if needed."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        nested_path = tmp_path / "nested" / "dir"
        dec_module.DATA_DIR = nested_path

        try:
            ctx = PatternContext(mock_state, "map", "test_pattern")
            test_data = {"result": "test"}

            output_path = ctx.save_output(test_data)

            # Parent directory should be created
            assert output_path.parent.exists()
            assert output_path.exists()
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_load_input(self, mock_state, tmp_path):
        """Test load_input reads pickle file."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            # Create input file
            input_data = {"circuit": "test_circuit", "observable": "ZZ"}
            input_path = tmp_path / "test_pattern_map_result.pkl"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, 'wb') as f:
                pickle.dump(input_data, f)

            # Load via context
            ctx = PatternContext(mock_state, "optimize", "test_pattern")
            loaded_data = ctx.load_input()

            assert loaded_data == input_data
        finally:
            dec_module.DATA_DIR = original_data_dir

    def test_load_input_raises_for_first_stage(self, mock_state):
        """Test load_input raises ValueError for map stage."""
        ctx = PatternContext(mock_state, "map", "test_pattern")

        with pytest.raises(ValueError, match="first stage"):
            ctx.load_input()

    def test_load_input_raises_for_missing_file(self, mock_state, tmp_path):
        """Test load_input raises FileNotFoundError if input doesn't exist."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            ctx = PatternContext(mock_state, "optimize", "test_pattern")

            with pytest.raises(FileNotFoundError, match="Input file not found"):
                ctx.load_input()
        finally:
            dec_module.DATA_DIR = original_data_dir


class TestPatternContextLogging:
    """Test logging functionality."""

    def test_log_basic(self, mock_state):
        """Test basic logging."""
        ctx = PatternContext(mock_state, "map", "test_pattern")

        ctx.log("Test message")

        logs = ctx.get_logs()
        assert len(logs) == 1
        assert "Test message" in logs[0]
        assert "[INFO]" in logs[0]
        assert "map:" in logs[0]

    def test_log_with_level(self, mock_state):
        """Test logging with different levels."""
        ctx = PatternContext(mock_state, "execute", "test_pattern")

        ctx.log("Info message", level="INFO")
        ctx.log("Warning message", level="WARNING")
        ctx.log("Error message", level="ERROR")

        logs = ctx.get_logs()
        assert len(logs) == 3
        assert "[INFO]" in logs[0]
        assert "[WARNING]" in logs[1]
        assert "[ERROR]" in logs[2]

    def test_get_logs_returns_copy(self, mock_state):
        """Test get_logs returns a copy of log list."""
        ctx = PatternContext(mock_state, "map", "test_pattern")
        ctx.log("Test message")

        logs1 = ctx.get_logs()
        logs2 = ctx.get_logs()

        # Should be equal but not the same object
        assert logs1 == logs2
        assert logs1 is not logs2

    def test_log_multiple_messages(self, mock_state):
        """Test logging multiple messages."""
        ctx = PatternContext(mock_state, "optimize", "test_pattern")

        ctx.log("Message 1")
        ctx.log("Message 2")
        ctx.log("Message 3")

        logs = ctx.get_logs()
        assert len(logs) == 3


class TestPatternContextConfig:
    """Test configuration access."""

    def test_get_config_existing_key(self, mock_state):
        """Test get_config returns value for existing key."""
        ctx = PatternContext(mock_state, "map", "test_pattern")

        value = ctx.get_config("custom_config")
        assert value == "test_value"

    def test_get_config_missing_key_with_default(self, mock_state):
        """Test get_config returns default for missing key."""
        ctx = PatternContext(mock_state, "map", "test_pattern")

        value = ctx.get_config("missing_key", default="default_value")
        assert value == "default_value"

    def test_get_config_missing_key_without_default(self, mock_state):
        """Test get_config returns None for missing key without default."""
        ctx = PatternContext(mock_state, "map", "test_pattern")

        value = ctx.get_config("missing_key")
        assert value is None

    def test_get_config_accesses_state(self, mock_state):
        """Test get_config accesses state dictionary."""
        mock_state["test_setting"] = 42
        ctx = PatternContext(mock_state, "map", "test_pattern")

        value = ctx.get_config("test_setting")
        assert value == 42


class TestPatternContextIntegration:
    """Test integrated workflows."""

    def test_full_stage_io_workflow(self, mock_state, tmp_path):
        """Test complete I/O workflow across stages."""
        import patterns.decorators as dec_module
        original_data_dir = dec_module.DATA_DIR
        dec_module.DATA_DIR = tmp_path

        try:
            # Map stage: create and save output
            map_ctx = PatternContext(mock_state, "map", "test_pattern")
            map_data = {"circuit": "bell_state", "parameter": "theta"}
            map_ctx.save_output(map_data)

            # Optimize stage: load map output, save new output
            opt_ctx = PatternContext(mock_state, "optimize", "test_pattern")
            loaded_map_data = opt_ctx.load_input()
            assert loaded_map_data == map_data

            opt_data = {"circuit": "optimized_bell", "depth": 3}
            opt_ctx.save_output(opt_data)

            # Execute stage: load optimize output
            exec_ctx = PatternContext(mock_state, "execute", "test_pattern")
            loaded_opt_data = exec_ctx.load_input()
            assert loaded_opt_data == opt_data

        finally:
            dec_module.DATA_DIR = original_data_dir
