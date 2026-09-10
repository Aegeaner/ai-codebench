"""Claude provider using the OpenAI-compatible API."""

from .openai_compatible import OpenAICompatibleProvider


class ClaudeProvider(OpenAICompatibleProvider):
    """Claude provider via Anthropic's OpenAI-compatibility layer."""
