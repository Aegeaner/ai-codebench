"""Configuration management for AI Chat Assistant"""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from ai_codebench.settings import (
    Settings,
    TaskType,
    Provider,
    DEFAULT_MODELS,
    IMAGE_MODELS,
    NANO_BANANA_MODELS,
    PROVIDER_MODEL_PATTERNS,
)


@dataclass
class ApplicationConfig:
    """Application configuration and provider routing."""

    settings: Settings

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "ApplicationConfig":
        """Load configuration from YAML file and environment variables."""
        return cls(settings=Settings.from_file(config_path))

    def resolve_provider_for_model(self, model_name: str) -> Optional[Provider]:
        """Resolve the provider from a model name using known name patterns."""
        model_lower = model_name.lower()
        for provider, patterns in PROVIDER_MODEL_PATTERNS.items():
            if any(model_lower.startswith(pattern) for pattern in patterns):
                return provider
        return None

    def get_model_for_provider_and_task(
        self, provider: Provider, task_type: TaskType
    ) -> str:
        """Get the configured model for a provider and task type."""
        provider_config = self.settings.provider_configs.get(provider)

        if task_type == TaskType.IMAGE:
            if provider_config and provider_config.image_model:
                return provider_config.image_model
            return IMAGE_MODELS.get(provider, DEFAULT_MODELS.get(provider, ""))

        if not provider_config:
            return DEFAULT_MODELS.get(provider, "")

        if task_type in (TaskType.KNOWLEDGE, TaskType.WRITE):
            return provider_config.knowledge_model
        if task_type == TaskType.CODE:
            return provider_config.code_model
        return provider_config.default_model

    def get_image_models(self, provider: Provider) -> List[str]:
        """Get available image generation models for a provider."""
        models = []
        provider_config = self.settings.provider_configs.get(provider)
        if provider_config and provider_config.image_model:
            models.append(provider_config.image_model)
        if provider in IMAGE_MODELS:
            models.append(IMAGE_MODELS[provider])
        if provider == Provider.GEMINI:
            models.extend(NANO_BANANA_MODELS)
        return list(dict.fromkeys(models))

    def has_api_key(self, provider: Provider) -> bool:
        """Check if credentials are available for a provider."""
        return self.settings.has_credentials(provider)

    @property
    def enable_async_answers(self) -> bool:
        """Whether answers are saved asynchronously in the background."""
        return self.settings.enable_async_answers

    @enable_async_answers.setter
    def enable_async_answers(self, value: bool) -> None:
        self.settings.enable_async_answers = value

    def get_available_providers(self) -> List[Provider]:
        """Get list of providers with available credentials."""
        return [provider for provider in Provider if self.has_api_key(provider)]
