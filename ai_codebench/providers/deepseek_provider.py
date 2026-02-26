"""DeepSeek provider using OpenAI-compatible API"""

from typing import Dict, Optional, Any
from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider with OpenAI-compatible interface"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        model = default_model or (config.get("default_model") if config else None)
        if not model:
            raise ValueError("No default model configured for DeepSeek provider")
        super().__init__(
            api_key,
            enable_caching=enable_caching,
            default_model=model,
            base_url=base_url,
        )

    @property
    def supports_caching(self) -> bool:
        return False  # DeepSeek doesn't support context caching yet

    def _apply_task_parameters(self, model: str, kwargs: Dict[str, Any]):
        """Apply task-specific parameters for DeepSeek-specific models"""
        super()._apply_task_parameters(model, kwargs)
        
        # Ensure extra_body exists and add DeepSeek specific parameters
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        
        if isinstance(kwargs["extra_body"], dict):
            kwargs["extra_body"].update({"thinking": {"type": "enabled"}})

    def _extract_usage_stats(self, response: object) -> Dict[str, int]:
        """Extract usage stats from DeepSeek API response"""
        if not hasattr(response, "usage"):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        usage = response.usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
