"""OpenAI-compatible provider base class"""

from typing import AsyncGenerator, List, Optional
import openai
from openai import APIStatusError, APITimeoutError, APIConnectionError
from .base import BaseProvider, Message, ChatResponse, ProviderAPIError


class OpenAICompatibleProvider(BaseProvider):
    """Base class for OpenAI-compatible providers"""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Standard async OpenAI-style chat completion"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]

        try:
            response = await self.async_client.chat.completions.create(
                model=model, messages=messages_dict, **kwargs
            )
            return ChatResponse(
                content=response.choices[0].message.content,
                usage=self._extract_usage_stats(response),
            )
        except (APIStatusError, APITimeoutError, APIConnectionError, Exception) as e:
            raise ProviderAPIError(f"OpenAI-compatible API error: {e}") from e

    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[dict, None]:
        """Standard OpenAI-style streaming"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]

        try:
            stream = await self.async_client.chat.completions.create(
                model=model, messages=messages_dict, stream=True, **kwargs
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {"text": chunk.choices[0].delta.content}
                if hasattr(chunk, "usage"):
                    # This is a simplification; actual usage might come at the end of the stream
                    # For now, we'll yield if available, but real-world might need aggregation
                    usage = self._extract_usage_stats(chunk)
                    if usage: # Only yield if there's actual usage data
                        yield {"usage": usage}
        except (APIStatusError, APITimeoutError, APIConnectionError, Exception) as e:
            yield {"error": f"OpenAI-compatible streaming API error: {e}"}

    @property
    def supports_caching(self) -> bool:
        return True

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_async_batch(self) -> bool:
        return True
