"""OpenRouter provider implementation"""

from typing import AsyncGenerator, Dict, List, Optional
from .openai_compatible import OpenAICompatibleProvider
from .base import Message, ChatResponse


class OpenRouterProvider(OpenAICompatibleProvider):
    """Provider for OpenRouter API"""

    def __init__(
        self,
        api_key: str,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key, base_url=base_url, **kwargs)
        self.referer = referer
        self.title = title

    def _get_extra_headers(self) -> Dict[str, str]:
        """Generate extra headers for OpenRouter API"""
        headers = {}
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """OpenRouter async chat completion with extra headers"""
        extra_headers = self._get_extra_headers()
        return await super().chat_completion(
            messages, model, extra_headers=extra_headers, **kwargs
        )

    def chat_completion_sync(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """OpenRouter sync chat completion with extra headers"""
        extra_headers = self._get_extra_headers()
        return super().chat_completion_sync(
            messages, model, extra_headers=extra_headers, **kwargs
        )

    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """OpenRouter streaming completion with extra headers"""
        extra_headers = self._get_extra_headers()
        async for chunk in super().stream_completion(
            messages, model, extra_headers=extra_headers, **kwargs
        ):
            yield chunk
