"""Kimi provider using the OpenAI-compatible API."""

from typing import Dict, Any
from .openai_compatible import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    """Kimi provider with OpenAI-compatible interface."""

    @property
    def supports_caching(self) -> bool:
        return False

    def _apply_task_parameters(self, model: str, kwargs: Dict[str, Any]):
        """Apply task-specific parameters for Kimi-specific models."""
        super()._apply_task_parameters(model, kwargs)

        if model in ["kimi-k2-thinking", "kimi-k2.5"]:
            # Kimi Thinking series strict requirements
            kwargs["temperature"] = 1.0
            kwargs["top_p"] = 0.95
            kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), 32768)
