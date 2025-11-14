"""OpenAI-compatible provider base class"""

from typing import AsyncGenerator, Dict, List, Optional
import openai
from .base import BaseProvider, Message, ChatResponse


class OpenAICompatibleProvider(BaseProvider):
    """Base class for OpenAI-compatible providers"""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
        self.sync_client = openai.OpenAI(api_key=api_key)

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Standard async OpenAI-style chat completion"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]

        response = await self.async_client.chat.completions.create(
            model=model, messages=messages_dict, **kwargs
        )

        return ChatResponse(
            content=response.choices[0].message.content,
            usage=self._extract_usage_stats(response),
        )

    def chat_completion_sync(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Standard sync OpenAI-style chat completion"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]

        response = self.sync_client.chat.completions.create(
            model=model, messages=messages_dict, **kwargs
        )

        return ChatResponse(
            content=response.choices[0].message.content,
            usage=self._extract_usage_stats(response),
        )

    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Standard OpenAI-style streaming"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]

        stream = await self.async_client.chat.completions.create(
            model=model, messages=messages_dict, stream=True, **kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield {"text": chunk.choices[0].delta.content}
            if hasattr(chunk, "usage"):
                yield {"usage": self._extract_usage_stats(chunk)}

    @property
    def supports_caching(self) -> bool:
        return True

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_async_batch(self) -> bool:
        return True
