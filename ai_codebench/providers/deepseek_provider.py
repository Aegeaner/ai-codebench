"""DeepSeek provider using the OpenAI-compatible API."""

from typing import Dict, Any
from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider with OpenAI-compatible interface."""

    @property
    def supports_caching(self) -> bool:
        return False

    def _apply_task_parameters(self, model: str, kwargs: Dict[str, Any]):
        """Apply task-specific parameters for DeepSeek-specific models."""
        super()._apply_task_parameters(model, kwargs)

        kwargs.setdefault("extra_body", {})
        if isinstance(kwargs["extra_body"], dict):
            kwargs["extra_body"].update({"thinking": {"type": "enabled"}})
