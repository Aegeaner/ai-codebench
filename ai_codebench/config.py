"""Configuration management for AI Chat Assistant"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
from ai_codebench.settings import (
    Settings,
    TaskType,
    Provider,
    DEFAULT_MODELS,
    IMAGE_MODELS,
)


@dataclass
class ApplicationConfig:
    """Application configuration and provider routing"""

    settings: Settings

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "ApplicationConfig":
        """Load configuration from YAML file and environment variables"""
        settings = Settings.from_file(config_path)
        return cls(settings=settings)

    def get_provider_for_task(self, task_type: TaskType) -> Provider:
        """Get the preferred provider for a given task type"""
        if task_type == TaskType.KNOWLEDGE:
            return self.settings.knowledge_provider
        elif task_type == TaskType.CODE:
            return self.settings.code_provider
        elif task_type == TaskType.IMAGE:
            return self.settings.image_provider
        else:
            return self.settings.knowledge_provider  # Fallback

    def get_model_for_provider_and_task(
        self, provider: Provider, task_type: TaskType
    ) -> str:
        """Get the configured model for a provider and task type"""
        provider_config = self.settings.provider_configs.get(provider)
        
        if task_type == TaskType.IMAGE:
            if provider_config and provider_config.image_model:
                return provider_config.image_model
            return IMAGE_MODELS.get(provider, DEFAULT_MODELS.get(provider, ""))

        if not provider_config:
            return DEFAULT_MODELS.get(provider, "")

        if task_type == TaskType.KNOWLEDGE or task_type == TaskType.WRITE:
            return provider_config.knowledge_model
        elif task_type == TaskType.CODE:
            return provider_config.code_model
        return provider_config.default_model

    def get_provider_models(self, provider: Provider) -> List[Dict]:
        """Get all configured models for a provider"""
        provider_config = self.settings.provider_configs.get(provider)
        if not provider_config:
            return []

        models = []
        if provider_config:
            all_models = {m["name"]: m for m in getattr(provider_config, "models", [])}

            for model_name in {
                provider_config.default_model,
                provider_config.knowledge_model,
                provider_config.code_model,
                provider_config.image_model,
            }:
                if model_name and model_name not in all_models:
                    all_models[model_name] = {"name": model_name}

            models = [
                {
                    "name": model["name"],
                    "supports_chat": model["name"] == provider_config.knowledge_model,
                    "supports_coding": model["name"] == provider_config.code_model,
                    "supports_image": model["name"] == provider_config.image_model,
                }
                for model in all_models.values()
            ]

        return models

    def get_default_model(self, provider: Provider) -> str:
        """Get the default model for a provider"""
        provider_config = self.settings.provider_configs.get(provider)
        if provider_config:
            return provider_config.default_model
        return DEFAULT_MODELS.get(provider, "")

    def has_api_key(self, provider: Provider) -> bool:
        """Check if API key is available for a provider"""
        key_map = {
            Provider.CLAUDE: self.settings.ANTHROPIC_API_KEY,
            Provider.DEEPSEEK: self.settings.deepseek_api_key,
            Provider.GEMINI: self.settings.gemini_api_key,
            Provider.OPENROUTER: self.settings.openrouter_api_key,
            Provider.KIMI: self.settings.kimi_api_key,
            Provider.HUNYUAN: self.settings.tencent_secret_id and self.settings.tencent_secret_key,
        }
        return key_map.get(provider) is not None

    def get_available_providers(self) -> List[Provider]:
        """Get list of providers with available API keys"""
        return [provider for provider in Provider if self.has_api_key(provider)]

    def get_fallback_providers(
        self, exclude: Optional[Provider] = None
    ) -> List[Provider]:
        """Get fallback providers in order, optionally excluding one"""
        available = self.get_available_providers()
        fallback = [p for p in self.settings.fallback_order if p in available]

        if exclude:
            fallback = [p for p in fallback if p != exclude]

        return fallback
