"""Claude provider using OpenAI-compatible API"""

from typing import Optional
from openai import AsyncOpenAI, OpenAI
from .openai_compatible import OpenAICompatibleProvider


class ClaudeProvider(OpenAICompatibleProvider):
    """Claude provider using OpenAI-compatible API"""

    def __init__(
        self,
        api_key: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            api_key,
            enable_caching=enable_caching,
            default_model=default_model,
            **kwargs,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url="https://api.anthropic.com/v1/"
        )
        self.sync_client = OpenAI(
            api_key=api_key, base_url="https://api.anthropic.com/v1/"
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
        return "claude-sonnet-4-20250514"
