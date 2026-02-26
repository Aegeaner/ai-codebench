"""Kimi provider using OpenAI-compatible API"""

from typing import Dict, Optional, Any
from .openai_compatible import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    """Kimi provider with OpenAI-compatible interface"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        model = default_model or (config.get("default_model") if config else None)
        if not model:
            raise ValueError("No default model configured for Kimi provider")
        super().__init__(
            api_key,
            enable_caching=enable_caching,
            default_model=model,
            base_url=base_url,
        )

    @property
    def supports_caching(self) -> bool:
        return False

    def _apply_task_parameters(self, model: str, kwargs: Dict[str, Any]):
        """Apply task-specific parameters for Kimi-specific models"""
        super()._apply_task_parameters(model, kwargs)

        if model in ["kimi-k2-thinking", "kimi-k2.5"]:
            # Kimi Thinking series strict requirements
            kwargs["temperature"] = 1.0
            kwargs["top_p"] = 0.95
            
            # Ensure sufficient max_tokens
            current_max = kwargs.get("max_tokens", 0)
            if current_max < 32768:
                kwargs["max_tokens"] = 32768
