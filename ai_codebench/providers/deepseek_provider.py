"""DeepSeek provider using OpenAI-compatible API"""

from typing import AsyncGenerator, Dict, List, Optional
from openai import AsyncOpenAI, OpenAI
from .openai_compatible import OpenAICompatibleProvider, Message, ChatResponse


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider with OpenAI-compatible interface"""

    def __init__(
        self,
        api_key: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        model = default_model or (config.get("default_model") if config else None)
        if not model:
            raise ValueError("No default model configured for DeepSeek provider")
        super().__init__(api_key, enable_caching=enable_caching, default_model=model)
        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url="https://api.deepseek.com/v1"
        )
        self.sync_client = OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com/v1"
        )

    @property
    def supports_caching(self) -> bool:
        return False  # DeepSeek doesn't support context caching yet

    def _extract_usage_stats(self, response: object) -> Dict[str, int]:
        """Extract usage stats from DeepSeek API response"""
        if not hasattr(response, "usage"):
            return None

        usage = response.usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[dict, None]:
        """Generate async streaming chat completion"""
        model = model or self.default_model
        messages_dict = [msg.to_dict() for msg in messages]
        cumulative_usage = None

        stream = await self.async_client.chat.completions.create(
            model=model, messages=messages_dict, stream=True, **kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_usage = self._extract_usage_stats(chunk) if hasattr(chunk, "usage") else None
                if chunk_usage:
                    if cumulative_usage is None:
                        cumulative_usage = chunk_usage
                    else:
                        cumulative_usage = {
                            "prompt_tokens": cumulative_usage["prompt_tokens"] + chunk_usage["prompt_tokens"],
                            "completion_tokens": cumulative_usage["completion_tokens"] + chunk_usage["completion_tokens"],
                            "total_tokens": cumulative_usage["total_tokens"] + chunk_usage["total_tokens"]
                        }
                yield {
                    "text": chunk.choices[0].delta.content,
                    "usage": chunk_usage
                }

        # Yield final cumulative usage if available
        if cumulative_usage:
            yield {"usage": cumulative_usage}
