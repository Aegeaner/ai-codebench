"""Task routing and provider selection logic"""

from typing import Optional, Dict, Any
from .config import Config, TaskType, Provider
from .providers import BaseProvider, ClaudeProvider, DeepSeekProvider, GeminiProvider


class TaskRouter:
    """Routes tasks to appropriate AI providers based on content analysis"""

    def __init__(self, config: Config):
        self.config = config
        self._providers: Dict[Provider, BaseProvider] = {}
        self.current_task_type = TaskType.KNOWLEDGE  # Default to knowledge tasks
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers based on API keys"""
        if self.config.has_api_key(Provider.CLAUDE):
            default_model = self.config.get_default_model(Provider.CLAUDE)
            self._providers[Provider.CLAUDE] = ClaudeProvider(
                self.config.ANTHROPIC_API_KEY,
                self.config.enable_context_caching,
                default_model,
            )

        if self.config.has_api_key(Provider.DEEPSEEK):
            default_model = self.config.get_default_model(Provider.DEEPSEEK)
            self._providers[Provider.DEEPSEEK] = DeepSeekProvider(
                self.config.deepseek_api_key,
                self.config.enable_context_caching,
                default_model,
            )

        if self.config.has_api_key(Provider.GEMINI):
            default_model = self.config.get_default_model(Provider.GEMINI)
            self._providers[Provider.GEMINI] = GeminiProvider(
                self.config.gemini_api_key,
                self.config.enable_context_caching,
                default_model,
            )

    def get_provider_for_task(
        self, task_type: TaskType, preferred_provider: Optional[Provider] = None
    ) -> BaseProvider:
        """Get the best available provider for a given task type"""

        # Use preferred provider if specified and available
        if preferred_provider and preferred_provider in self._providers:
            return self._providers[preferred_provider]

        # Get default provider for task type
        default_provider = self.config.get_provider_for_task(task_type)

        # Return default if available
        if default_provider in self._providers:
            return self._providers[default_provider]

        # Fallback to any available provider
        if self._providers:
            return next(iter(self._providers.values()))

        raise RuntimeError("No AI providers available. Please check your API keys.")

    def get_available_providers(self) -> Dict[Provider, BaseProvider]:
        """Get all available providers"""
        return self._providers.copy()

    def get_provider_info(self, provider: Provider) -> Dict[str, Any]:
        """Get information about a specific provider"""
        if provider not in self._providers:
            return {"available": False}

        provider_instance = self._providers[provider]
        return {
            "available": True,
            "default_model": provider_instance.default_model,
            "supports_caching": provider_instance.supports_caching,
            "supports_async_batch": provider_instance.supports_async_batch,
        }
