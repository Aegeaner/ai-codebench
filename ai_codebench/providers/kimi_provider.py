"""Kimi provider using OpenAI-compatible API"""

from typing import Dict, Optional
from .openai_compatible import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    """Kimi provider with OpenAI-compatible interface"""

    def __init__(
        self,
        api_key: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        model = default_model or (config.get("default_model") if config else None)
        if not model:
            raise ValueError("No default model configured for Kimi provider")
        super().__init__(api_key, enable_caching=enable_caching, default_model=model)
        self.async_client.base_url = "https://api.moonshot.cn/v1"

    @property
    def supports_caching(self) -> bool:
        return False
