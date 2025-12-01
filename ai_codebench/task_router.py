"""Task routing and provider selection logic"""

from typing import Optional, Dict, Any, List
from .config import ApplicationConfig
from .provider_manager import ProviderManager
from .providers import BaseProvider
from .settings import TaskType, Provider


class TaskRouter:
    """Routes tasks to appropriate AI providers based on content analysis"""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.provider_manager = ProviderManager(config.settings)

    def get_provider_for_task(
        self, task_type: TaskType, preferred_provider: Optional[Provider] = None
    ) -> BaseProvider:
        """Get the best available provider for a given task type"""

        # Use preferred provider if specified and available
        if preferred_provider:
            provider_instance = self.provider_manager.get_provider(
                preferred_provider, task_type=task_type
            )
            if provider_instance:
                return provider_instance

        # Get default provider for task type
        default_provider_type = self.config.get_provider_for_task(task_type)
        default_provider_instance = self.provider_manager.get_provider(
            default_provider_type, task_type=task_type
        )

        if default_provider_instance:
            return default_provider_instance

        # Fallback to any available provider
        available_provider_types = self.provider_manager.get_available_providers()
        for provider_type in self.config.settings.fallback_order:
            if provider_type in available_provider_types:
                fallback_provider_instance = self.provider_manager.get_provider(
                    provider_type, task_type=task_type
                )
                if fallback_provider_instance:
                    return fallback_provider_instance

        raise RuntimeError("No AI providers available. Please check your API keys.")

    def get_available_providers(self) -> Dict[Provider, BaseProvider]:
        """Get all available providers"""
        available_types = self.provider_manager.get_available_providers()
        return {
            p_type: self.provider_manager.get_provider(p_type)
            for p_type in available_types
            if self.provider_manager.get_provider(p_type) is not None
        }

    def get_provider_info(self, provider: Provider) -> Dict[str, Any]:
        """Get information about a specific provider"""
        provider_instance = self.provider_manager.get_provider(provider)
        if not provider_instance:
            return {"available": False}

        return {
            "available": True,
            "default_model": provider_instance.default_model,
            "supports_caching": provider_instance.supports_caching,
            "supports_async_batch": provider_instance.supports_async_batch,
        }
