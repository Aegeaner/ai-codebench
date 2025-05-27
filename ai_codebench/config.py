"""Configuration management for AI Chat Assistant"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
from enum import Enum


class TaskType(Enum):
    KNOWLEDGE = "knowledge"
    CODE = "code"


class Provider(Enum):
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


# Default models for each provider
DEFAULT_MODELS = {
    Provider.CLAUDE: "claude-sonnet-4-20250514",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-2.5-flash-preview-05-20",
}


@dataclass
class ProviderConfig:
    """Configuration for a specific provider"""

    default_model: str
    knowledge_model: str
    code_model: str
    models: List[Dict] = field(default_factory=list)


@dataclass
class Config:
    """Application configuration"""

    # API Keys from environment
    ANTHROPIC_API_KEY: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Chat history settings
    history_window_size: int = 3

    # Model selection preferences
    knowledge_provider: Provider = Provider.CLAUDE
    code_provider: Provider = Provider.DEEPSEEK
    fallback_order: List[Provider] = field(
        default_factory=lambda: [Provider.CLAUDE, Provider.DEEPSEEK, Provider.GEMINI]
    )

    # Performance settings
    enable_context_caching: bool = True
    enable_async_calls: bool = True

    # Provider configurations
    provider_configs: Dict[Provider, ProviderConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Load configuration from environment variables and config file"""
        self._load_api_keys()
        self._load_from_env()

    def _load_api_keys(self):
        """Load API keys from environment variables"""
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def _load_from_env(self):
        """Override defaults from environment variables"""
        if os.getenv("HISTORY_WINDOW_SIZE"):
            self.history_window_size = int(os.getenv("HISTORY_WINDOW_SIZE"))

    def _get_provider_config(self, provider: Provider) -> Optional[ProviderConfig]:
        """Helper to get provider config with fallback to defaults"""
        if provider in self.provider_configs:
            return self.provider_configs[provider]
        return None

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file"""
        if config_path is None:
            # Try default locations
            default_paths = [
                Path.cwd() / "config.yaml",
                Path.cwd() / "config.yml",
                Path.home() / ".ai_codebench_config.yaml",
                Path.home() / ".ai_codebench_config.yml",
            ]

            config_path = None
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        config = cls()

        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)

                config._load_from_yaml(yaml_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")

        return config

    def _load_from_yaml(self, yaml_config: Dict):
        """Load configuration from parsed YAML"""

        # Conversation settings
        if "conversation" in yaml_config:
            conv_config = yaml_config["conversation"]
            if "history_window_size" in conv_config:
                self.history_window_size = conv_config["history_window_size"]

        # Performance settings
        if "performance" in yaml_config:
            perf_config = yaml_config["performance"]
            if "enable_context_caching" in perf_config:
                self.enable_context_caching = perf_config["enable_context_caching"]
            if "enable_async_calls" in perf_config:
                self.enable_async_calls = perf_config["enable_async_calls"]

        # Provider configurations
        if "providers" in yaml_config:
            providers_config = yaml_config["providers"]

            for provider_name, provider_data in providers_config.items():
                try:
                    provider = Provider(provider_name)
                    self.provider_configs[provider] = ProviderConfig(
                        default_model=provider_data.get("default_model", ""),
                        knowledge_model=provider_data.get(
                            "knowledge_model", provider_data.get("default_model", "")
                        ),
                        code_model=provider_data.get(
                            "code_model", provider_data.get("default_model", "")
                        ),
                        models=provider_data.get("models", []),
                    )
                except ValueError:
                    print(f"Warning: Unknown provider in config: {provider_name}")

        # Task routing preferences
        if "task_routing" in yaml_config:
            routing_config = yaml_config["task_routing"]

            if "knowledge_provider" in routing_config:
                try:
                    self.knowledge_provider = Provider(
                        routing_config["knowledge_provider"]
                    )
                except ValueError:
                    print(
                        f"Warning: Unknown knowledge provider: {routing_config['knowledge_provider']}"
                    )

            if "code_provider" in routing_config:
                try:
                    self.code_provider = Provider(routing_config["code_provider"])
                except ValueError:
                    print(
                        f"Warning: Unknown code provider: {routing_config['code_provider']}"
                    )

            if "fallback_order" in routing_config:
                fallback_list = []
                for provider_name in routing_config["fallback_order"]:
                    try:
                        fallback_list.append(Provider(provider_name))
                    except ValueError:
                        print(f"Warning: Unknown fallback provider: {provider_name}")
                if fallback_list:
                    self.fallback_order = fallback_list

    def get_provider_for_task(self, task_type: TaskType) -> Provider:
        """Get the preferred provider for a given task type"""
        if task_type == TaskType.KNOWLEDGE:
            return self.knowledge_provider
        elif task_type == TaskType.CODE:
            return self.code_provider
        else:
            return self.knowledge_provider  # Fallback

    def get_model_for_provider_and_task(
        self, provider: Provider, task_type: TaskType
    ) -> str:
        """Get the configured model for a provider and task type"""
        provider_config = self._get_provider_config(provider)
        if not provider_config:
            return DEFAULT_MODELS.get(provider, "")

        if task_type == TaskType.KNOWLEDGE:
            return provider_config.knowledge_model
        elif task_type == TaskType.CODE:
            return provider_config.code_model
        return provider_config.default_model

    def get_provider_models(self, provider: Provider) -> List[Dict]:
        """Get all configured models for a provider"""
        provider_config = self.provider_configs.get(provider)
        if not provider_config:
            return []

        # Get models from YAML configuration
        models = []
        if provider_config:
            # Include default models if not explicitly listed
            all_models = {m["name"]: m for m in getattr(provider_config, "models", [])}

            # Ensure default/knowledge/code models are included
            for model_name in {
                provider_config.default_model,
                provider_config.knowledge_model,
                provider_config.code_model,
            }:
                if model_name and model_name not in all_models:
                    all_models[model_name] = {"name": model_name}

            models = [
                {
                    "name": model["name"],
                    "supports_chat": model["name"] == provider_config.knowledge_model,
                    "supports_coding": model["name"] == provider_config.code_model,
                }
                for model in all_models.values()
            ]

        return models

    def get_default_model(self, provider: Provider) -> str:
        """Get the default model for a provider"""
        provider_config = self._get_provider_config(provider)
        if provider_config:
            return provider_config.default_model
        return DEFAULT_MODELS.get(provider, "")

    def has_api_key(self, provider: Provider) -> bool:
        """Check if API key is available for a provider"""
        key_map = {
            Provider.CLAUDE: self.ANTHROPIC_API_KEY,
            Provider.DEEPSEEK: self.deepseek_api_key,
            Provider.GEMINI: self.gemini_api_key,
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
        fallback = [p for p in self.fallback_order if p in available]

        if exclude:
            fallback = [p for p in fallback if p != exclude]

        return fallback
