"""Manages the instantiation and retrieval of AI model providers."""

from typing import Dict, Optional, List

from ai_codebench.providers.base import BaseProvider
from ai_codebench.providers.claude_provider import ClaudeProvider
from ai_codebench.providers.deepseek_provider import DeepSeekProvider
from ai_codebench.providers.gemini_provider import GeminiProvider
from ai_codebench.providers.kimi_provider import KimiProvider
from ai_codebench.providers.openai_compatible import OpenAICompatibleProvider
from ai_codebench.settings import Provider, Settings, DEFAULT_MODELS


class ProviderManager:
    """Manages the instantiation and retrieval of AI model providers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers: Dict[Provider, BaseProvider] = {}

    def _get_default_model(self, provider: Provider) -> str:
        """Get the default model for a provider from settings."""
        provider_config = self.settings.provider_configs.get(provider)
        if provider_config and provider_config.default_model:
            return provider_config.default_model
        return DEFAULT_MODELS.get(provider, "")

    def _create_provider(self, provider_type: Provider) -> Optional[BaseProvider]:
        """Internal method to create a provider instance."""
        default_model = self._get_default_model(provider_type)
        if provider_type == Provider.CLAUDE and self.settings.ANTHROPIC_API_KEY:
            return ClaudeProvider(api_key=self.settings.ANTHROPIC_API_KEY, default_model=default_model)
        elif provider_type == Provider.DEEPSEEK and self.settings.deepseek_api_key:
            return DeepSeekProvider(api_key=self.settings.deepseek_api_key, default_model=default_model)
        elif provider_type == Provider.GEMINI and self.settings.gemini_api_key:
            return GeminiProvider(api_key=self.settings.gemini_api_key, default_model=default_model)
        elif provider_type == Provider.OPENROUTER and self.settings.openrouter_api_key:
            return OpenAICompatibleProvider(api_key=self.settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1", default_model=default_model)
        elif provider_type == Provider.KIMI and self.settings.kimi_api_key:
            return KimiProvider(api_key=self.settings.kimi_api_key, default_model=default_model)
        return None

    def get_provider(self, provider_type: Provider) -> Optional[BaseProvider]:
        """Get a provider instance, creating it if it doesn't already exist."""
        if provider_type not in self._providers:
            provider = self._create_provider(provider_type)
            if provider:
                self._providers[provider_type] = provider
        return self._providers.get(provider_type)

    def has_api_key(self, provider: Provider) -> bool:
        """Check if API key is available for a provider"""
        key_map = {
            Provider.CLAUDE: self.settings.ANTHROPIC_API_KEY,
            Provider.DEEPSEEK: self.settings.deepseek_api_key,
            Provider.GEMINI: self.settings.gemini_api_key,
            Provider.OPENROUTER: self.settings.openrouter_api_key,
            Provider.KIMI: self.settings.kimi_api_key,
        }
        return key_map.get(provider) is not None

    def get_available_providers(self) -> List[Provider]:
        """Get list of providers with available API keys"""
        return [provider for provider in Provider if self.has_api_key(provider)]

