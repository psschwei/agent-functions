"""
LLM client for orchestrator reasoning.

Provides OpenAI-compatible interface supporting litellm proxy and direct endpoints.
"""
import os
from typing import Optional, Dict, Any, List
from openai import OpenAI


class LLMClient:
    """
    OpenAI-compatible LLM client for orchestrator reasoning.

    Supports litellm proxy and direct OpenAI/Anthropic endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize LLM client.

        Args:
            model: Model name (e.g., "gpt-4", "claude-sonnet-4-5")
            base_url: Base URL for litellm proxy (e.g., "http://localhost:4000")
            api_key: API key (falls back to OPENAI_API_KEY environment variable)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize OpenAI client
        # If base_url is None, it will use OpenAI's default endpoint
        # If base_url is set (e.g., litellm proxy), it will use that instead
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "dummy-key"),
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """
        Call LLM with chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Tool selection strategy ('auto', 'required', 'none')

        Returns:
            Response dict with 'content', 'tool_calls', 'finish_reason', 'usage'
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
