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
    WRITE = "write"


class Provider(Enum):
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    KIMI = "kimi"


# Task-specific generation parameters
TASK_GENERATION_CONFIG = {
    TaskType.CODE: {"temperature": 0.0, "top_p": 1.0, "top_k": 50},
    TaskType.KNOWLEDGE: {"temperature": 0.4, "top_p": 0.95, "top_k": 50},
    TaskType.WRITE: {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
}


# System prompts for different task types
SYSTEM_PROMPTS = {
    TaskType.KNOWLEDGE: "You are a helpful AI assistant for knowledge learning. Teach the concept step by step.",
    TaskType.CODE: "You are a helpful AI assistant for code tasks. Analyze the algorithm ideas, algorithm steps and computational complexity, but don't write specific code. Please respond in Simplified Chinese.",
    TaskType.WRITE: "You are a helpful AI assistant for writing instruction, who polishes English drafts for clarity, grammar, and natural tone. Keep the author's voice as much as possible.",
    "default": "You are a helpful AI assistant for both knowledge learning and code tasks. For the knowledge learning task, teach the concept step by step. For the code tasks, you only need to analyze the algorithm ideas, algorithm steps and computational complexity, but don't write specific code.",
}


# Default models for each provider
DEFAULT_MODELS = {
    Provider.CLAUDE: "claude-sonnet-4-5-20250929",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-flash-latest",
    Provider.OPENROUTER: "openai/gpt-4o",
    Provider.KIMI: "kimi-k2-thinking",
}


@dataclass
class ProviderConfig:
    """Configuration for a specific provider"""

    default_model: str
    knowledge_model: str
    code_model: str
    base_url: str
    max_tokens: Optional[int] = None
    models: List[Dict] = field(default_factory=list)


@dataclass
class Settings:
    """Application settings loaded from environment variables and config file"""

    # API Keys from environment
    ANTHROPIC_API_KEY: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Chat history settings
    history_window_size: int = 3

    # Model selection preferences
    knowledge_provider: Provider = Provider.CLAUDE
    code_provider: Provider = Provider.DEEPSEEK
    fallback_order: List[Provider] = field(
        default_factory=lambda: [
            Provider.CLAUDE,
            Provider.DEEPSEEK,
            Provider.GEMINI,
            Provider.KIMI,
        ]
    )

    # Performance settings
    enable_context_caching: bool = True
    enable_async_calls: bool = True
    enable_async_answers: bool = False

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
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.kimi_api_key = os.getenv("MOONSHOT_API_KEY")

    def _load_from_env(self):
        """Override defaults from environment variables"""
        history_size = os.getenv("HISTORY_WINDOW_SIZE")
        if history_size and history_size.isdigit():
            try:
                self.history_window_size = int(history_size)
            except (TypeError, ValueError):
                # Handle invalid values gracefully
                pass

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "Settings":
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

        settings = cls()

        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)

                settings._load_from_yaml(yaml_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")

        return settings

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
            if "enable_async_answers" in perf_config:
                self.enable_async_answers = perf_config["enable_async_answers"]

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
                        base_url=provider_data["base_url"],
                        max_tokens=provider_data.get("max_tokens"),
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