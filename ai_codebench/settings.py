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
    IMAGE = "image"


class Provider(Enum):
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    KIMI = "kimi"
    HUNYUAN = "hunyuan"
    IMAGEN = "imagen"


# Task-specific generation parameters
TASK_GENERATION_CONFIG = {
    TaskType.CODE: {"temperature": 0.0, "top_p": 1.0, "top_k": 50},
    TaskType.KNOWLEDGE: {"temperature": 0.3, "top_p": 0.9, "top_k": 40},
    TaskType.WRITE: {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
    TaskType.IMAGE: {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
}

# Image generation specific configuration
IMAGE_GENERATION_CONFIG = {
    "image_size": "2K",
    "aspect_ratio": "3:4",
}


# Task-specific system prompts
KNOWLEDGE_PROMPT = (
    "You are an expert polymath and tutor dedicated to deep conceptual mastery. Your goal is to guide the user toward a "
    "'First Principles' understanding. When explaining concepts, go beyond surface-level steps.\n\n"
    "1. Start with the theoretical foundation and the 'why' behind the concept.\n"
    "2. Use heuristics, analogies, and mental models to build intuition.\n"
    "3. Break down the process step-by-step with clear reasoning.\n"
    "4. Discuss edge cases, implications, or deeper connections to other fields to foster comprehensive understanding.\n\n"
    "5. Objective Voice: Maintain an authoritative, analytical tone. Strictly avoid self-introductions, conversational "
    "fillers, or first-person pronouns."
)

CODE_PROMPT = (
    "You are a Principal Software Engineer and Algorithm Specialist. Your approach is 'Architecture First.'\n\n"
    "No Persona: Do not introduce yourself or use phrases like \"As an engineer\" or \"I suggest.\" Start directly with "
    "the problem analysis.\n\n"
    "Deep Analysis: Before proposing solutions, analyze the problem's constraints and potential pitfalls.\n\n"
    "Algorithmic Strategy: Deeply analyze the core algorithmic idea. Explain the choice of paradigm and why this "
    "specific approach is optimal for the given constraints.\n\n"
    "Complexity & Trade-offs: Provide a clear breakdown of Big O time and space complexity. Compare the chosen "
    "strategy against potential alternatives.\n\n"
    "Adaptive Logic Flow: Outline the Step-by-step Logical Flow. Scale the structural detail to the "
    "problem's complexity:\n\n"
    "For straightforward algorithms, provide a streamlined sequential breakdown;\n"
    "Only introduce Modular Decomposition if the problem involves multiple distinct responsibilities or high "
    "architectural complexity. Avoid unnecessary abstraction for simple tasks.\n\n"
    "Response Language: Please provide all explanations and analysis in Simplified Chinese.\n\n"
    "Professionalism: The response should be formatted as a publication-ready technical article using standard "
    "industry terminology."
)

WRITE_PROMPT = (
    "You are a high-end developmental editor. Your mission is to transform drafts into polished, high-impact prose.\n\n"
    "Precision & Flow: Enhance vocabulary and sentence rhythm while maintaining a natural, sophisticated tone.\n\n"
    "Voice Preservation: You are a silent partner; strictly preserve the author’s original intent, 'soul,' and unique "
    "voice. Do not over-sanitize.\n\n"
    "Structural Insight: If the narrative logic or impact can be improved by reordering or cutting, provide a "
    "'Rationale' section explaining your editorial choices.\n\n"
    "Correction: Fix all grammatical errors and subtle linguistic nuances without being pedantic."
)

IMAGE_PROMPT = (
    "You are an image generation model specialized in creating visually striking, social-media-ready images.\n\n"
    "Generate a high-fidelity scene optimized for social media platforms.\n\n"
    "The visual style should be cinematic photography, with a natural, premium aesthetic and a sense of realism.\n\n"
    "Composition: clean, balanced, and immediately eye-catching, with a clear focal point and strong visual "
    "hierarchy that reads well on small screens.\n\n"
    "Technical quality: ultra-high detail equivalent to 2K resolution, natural high-end lighting, shallow depth of "
    "field where appropriate, and professional cinematic color grading.\n\n"
    "CRUCIAL CONSTRAINT: The image must contain absolutely NO text of any kind — no letters, no words, no numbers, "
    "no symbols, no logos, no captions, no watermarks, and no UI elements.\n\n"
    "Ensure that all surfaces, backgrounds, clothing, props, and objects are purely visual and completely free of "
    "any typography, signage, branding, or readable characters."
)

DEFAULT_PROMPT = (
    "You are a versatile AI assistant dedicated to deep conceptual mastery, "
    "technical excellence, and refined creative impact."
)

# System prompts for different task types
SYSTEM_PROMPTS = {
    TaskType.KNOWLEDGE: KNOWLEDGE_PROMPT,
    TaskType.CODE: CODE_PROMPT,
    TaskType.WRITE: WRITE_PROMPT,
    TaskType.IMAGE: IMAGE_PROMPT,
    "default": DEFAULT_PROMPT,
}


# Default models for each provider
DEFAULT_MODELS = {
    Provider.CLAUDE: "claude-sonnet-4-5-20250929",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-flash-latest",
    Provider.OPENROUTER: "openai/gpt-4o",
    Provider.KIMI: "kimi-k2-thinking",
    Provider.IMAGEN: "imagen-4.0-generate-001",
}

# Image models for each provider (if supported)
IMAGE_MODELS = {
    Provider.GEMINI: "gemini-3-pro-image-preview",
    Provider.HUNYUAN: "hunyuan-3.0",
}

# Supported Imagen models for Gemini
GEMINI_IMAGEN_MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
]

# Patterns to identify provider by model name
PROVIDER_MODEL_PATTERNS = {
    Provider.CLAUDE: ["claude-"],
    Provider.DEEPSEEK: ["deepseek-"],
    Provider.GEMINI: ["gemini-"],
    Provider.IMAGEN: ["imagen-"],
    Provider.KIMI: ["moonshot-"],
    Provider.HUNYUAN: ["hunyuan-"],
}

@dataclass
class ProviderConfig:
    """Configuration for a specific provider"""

    default_model: str
    knowledge_model: str
    code_model: str
    base_url: str
    image_model: str = ""
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
    tencent_secret_id: Optional[str] = None
    tencent_secret_key: Optional[str] = None

    # Chat history settings
    history_window_size: int = 3

    # Model selection preferences
    knowledge_provider: Provider = Provider.CLAUDE
    code_provider: Provider = Provider.DEEPSEEK
    image_provider: Provider = Provider.GEMINI
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
        self.tencent_secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
        self.tencent_secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")

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
                        image_model=provider_data.get("image_model", ""),
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

            if "image_provider" in routing_config:
                try:
                    self.image_provider = Provider(routing_config["image_provider"])
                except ValueError:
                    print(
                        f"Warning: Unknown image provider: {routing_config['image_provider']}"
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