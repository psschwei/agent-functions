"""
Logging utilities for agent-functions.

Provides console logging with stage markers and formatted output.
"""
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up console logging with formatted output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("agent-functions")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def print_banner(title: str, width: int = 60):
    """
    Print a banner with the given title.

    Args:
        title: Title text to display
        width: Width of the banner
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_stage_start(stage_name: str):
    """
    Print a stage start marker.

    Args:
        stage_name: Name of the stage
    """
    print_banner(f"Starting: {stage_name}")


def print_stage_end(stage_name: str, success: bool = True, duration: Optional[float] = None):
    """
    Print a stage end marker.

    Args:
        stage_name: Name of the stage
        success: Whether the stage completed successfully
        duration: Optional duration in seconds
    """
    status = "Completed" if success else "Failed"
    title = f"{status}: {stage_name}"

    if duration is not None:
        title += f" ({duration:.2f}s)"

    print_banner(title)


def print_section(title: str):
    """
    Print a section header.

    Args:
        title: Section title
    """
    print(f"\n{title}")
    print("-" * len(title))


def print_key_value(key: str, value: str, indent: int = 2):
    """
    Print a key-value pair with formatting.

    Args:
        key: Key name
        value: Value to display
        indent: Number of spaces to indent
    """
    print(f"{' ' * indent}{key}: {value}")


def print_status(message: str, success: bool = True):
    """
    Print a status message with a checkmark or X.

    Args:
        message: Status message
        success: Whether the status is successful
    """
    symbol = "✓" if success else "✗"
    print(f"{symbol} {message}")
