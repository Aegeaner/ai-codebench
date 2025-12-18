"""OpenAI-compatible provider base class"""

from typing import AsyncGenerator, List, Optional, Dict, Any
import openai
from openai import APIStatusError, APITimeoutError, APIConnectionError
from .base import BaseProvider, Message, ChatResponse, ProviderAPIError
from ..settings import TASK_GENERATION_CONFIG


class OpenAICompatibleProvider(BaseProvider):
    """Base class for OpenAI-compatible providers"""

    def __init__(self, api_key: str, base_url: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _apply_task_parameters(self, kwargs: Dict[str, Any]):
        """Apply task-specific parameters like temperature, top_p, top_k"""
        task = kwargs.pop("task", None)
        if not task:
            return

        config = TASK_GENERATION_CONFIG.get(task)
        if config:
            kwargs.setdefault("temperature", config["temperature"])
            kwargs.setdefault("top_p", config["top_p"])
            
            # Use extra_body for top_k as it's not a standard parameter
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            
            if isinstance(kwargs["extra_body"], dict):
                kwargs["extra_body"].setdefault("top_k", config["top_k"])

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Standard async OpenAI-style chat completion"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]

        self._apply_task_parameters(kwargs)

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

        self._apply_task_parameters(kwargs)

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
                    if usage:  # Only yield if there's actual usage data
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
