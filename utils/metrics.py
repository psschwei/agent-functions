"""
Metrics utilities for tracking execution timing and performance.
"""
import time
from typing import Dict, Any, Callable
from functools import wraps


class MetricsTracker:
    """Track timing metrics for workflow execution."""

    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, float] = {}

    def start_timer(self, name: str):
        """
        Start a timer for a named operation.

        Args:
            name: Name of the operation to time
        """
        self.start_times[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record the duration.

        Args:
            name: Name of the operation

        Returns:
            Duration in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")

        end_time = time.perf_counter()
        duration = end_time - self.start_times[name]

        self.metrics[name] = duration
        del self.start_times[name]

        return duration

    def get_metric(self, name: str) -> float:
        """
        Get a recorded metric.

        Args:
            name: Name of the metric

        Returns:
            Metric value in seconds
        """
        return self.metrics.get(name, 0.0)

    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all recorded metrics.

        Returns:
            Dictionary of metric name to value
        """
        return self.metrics.copy()

    def display_summary(self):
        """Display a summary of all metrics."""
        print("\n" + "=" * 60)
        print("Timing Metrics Summary")
        print("=" * 60)

        if not self.metrics:
            print("No metrics recorded")
            return

        total_time = sum(self.metrics.values())

        for name, duration in self.metrics.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"  {name}: {duration:.3f}s ({percentage:.1f}%)")

        print(f"\n  Total: {total_time:.3f}s")
        print("=" * 60)


def timing_decorator(metric_name: str = None):
    """
    Decorator to automatically time function execution.

    Args:
        metric_name: Optional name for the metric (uses function name if not provided)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or func.__name__
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                print(f"[Timing] {name}: {duration:.3f}s")

        return wrapper
    return decorator


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"
