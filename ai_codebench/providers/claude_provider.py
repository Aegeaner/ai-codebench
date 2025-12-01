"""Claude provider using OpenAI-compatible API"""

from typing import Optional
from .openai_compatible import OpenAICompatibleProvider


class ClaudeProvider(OpenAICompatibleProvider):
    """Claude provider using OpenAI-compatible API"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            api_key,
            enable_caching=enable_caching,
            default_model=default_model,
            base_url=base_url,  # Pass base_url directly
            **kwargs,
        )

    @property
    def supports_caching(self) -> bool:
        return True

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_async_batch(self) -> bool:
        return True

    def _get_fallback_default_model(self) -> str:
        return "claude-sonnet-4-5-20250929"
