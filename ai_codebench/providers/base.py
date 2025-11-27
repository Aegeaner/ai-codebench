"""Base provider interface"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass


@dataclass
class Message:
    """Chat message representation"""

    role: str  # "user", "assistant", "system"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Response from chat completion"""

    content: str
    usage: Optional[Dict[str, int]] = None
    cached: bool = False


class ProviderAPIError(Exception):
    """Custom exception for provider API errors"""
    pass


class BaseProvider(ABC):
    """Base class for AI providers"""

    def __init__(
        self,
        api_key: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
    ):
        """Initialize provider with common configuration"""
        self.api_key = api_key
        self.enable_caching = enable_caching
        self._configured_default_model = default_model

    @abstractmethod
    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Generate async chat completion response"""
        pass

    @property
    @abstractmethod
    def supports_async(self) -> bool:
        """Whether this provider supports async operations"""
        pass

    @abstractmethod
    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[dict, None]:
        """Generate async streaming chat completion
        
        Yields:
            dict: Contains either:
                - 'text': str - The text chunk
                - 'usage': dict - Usage statistics (optional)
                - 'error': str - Error message (if error occurs)
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def supports_caching(self) -> bool:
        """Whether this provider supports context caching"""
        pass

    @property
    @abstractmethod
    def supports_async_batch(self) -> bool:
        """Whether this provider supports async/batch operations"""
        pass

    @property
    def default_model(self) -> str:
        """Default model for this provider"""
        if self._configured_default_model:
            return self._configured_default_model
        raise ValueError("No default model configured for provider")

    def _extract_usage_stats(self, response: object) -> Dict[str, int]:
        """Standardized usage stats extraction across all providers"""
        if hasattr(response, "usage"):  # OpenAI/Anthropic/DeepSeek format
            usage = response.usage
            return {
                "prompt_tokens": getattr(
                    usage, "prompt_tokens", getattr(usage, "input_tokens", 0)
                ),
                "completion_tokens": getattr(
                    usage, "completion_tokens", getattr(usage, "output_tokens", 0)
                ),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        elif hasattr(response, "usage_metadata"):  # Gemini format
            return {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
