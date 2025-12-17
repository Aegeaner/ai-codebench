"""Google Gemini provider using the new google-genai SDK"""

import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any
from google import genai
from google.genai import types
from .base import BaseProvider, Message, ChatResponse, ProviderAPIError
from ..settings import TaskType, TASK_GENERATION_CONFIG


class GeminiProvider(BaseProvider):
    """Google Gemini provider"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
    ):
        super().__init__(api_key, enable_caching, default_model)
        self.base_url = base_url
        
        # Parse base_url and api_version for the new SDK
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        
        # If the path contains a version (e.g., /v1beta), extract it
        path_parts = [p for p in parsed.path.split('/') if p]
        api_version = None
        # Root URL without the version path
        clean_base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else base_url
        
        if path_parts and (path_parts[-1].startswith('v1') or path_parts[-1].startswith('v2')):
            api_version = path_parts[-1]
        else:
            # If no version detected in path, use the provided base_url as is
            clean_base_url = base_url
            
        http_options = {'base_url': clean_base_url}
        if api_version:
            http_options['api_version'] = api_version

        # Initialize the new Google Gen AI client
        self.client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )

    def _apply_task_parameters(self, kwargs: Dict[str, Any], model_name: str):
        """Apply task-specific parameters like temperature, top_p, top_k, and thinking_level"""
        task = kwargs.pop("task", None)
        if not task:
            return

        # Set thinking_level based on task for Gemini 3 models
        is_gemini_3 = "gemini-3" in model_name.lower()
        if is_gemini_3:
            if task == TaskType.CODE:
                kwargs["thinking_level"] = "high"
            else:
                kwargs["thinking_level"] = "low"

        config = TASK_GENERATION_CONFIG.get(task)
        if config:
            # For Gemini 3, we strongly recommend keeping the temperature parameter at its default value of 1.0.
            # While previous models often benefited from tuning temperature to control creativity versus determinism, Gemini 3's reasoning capabilities are optimized for the default setting. Changing the temperature (setting it below 1.0) may lead to unexpected behavior, such as looping or degraded performance, particularly in complex mathematical or reasoning tasks.
            if is_gemini_3:
                kwargs.setdefault("temperature", 1.0)
            else:
                kwargs.setdefault("temperature", config["temperature"])
                
            kwargs.setdefault("top_p", config["top_p"])
            kwargs.setdefault("top_k", config["top_k"])

    def _get_generation_config(self, model_name: str, **kwargs) -> types.GenerateContentConfig:
        """Create GenerateContentConfig from kwargs"""
        is_gemini_3 = "gemini-3" in model_name.lower()
        
        thinking_config = None
        if is_gemini_3:
             thinking_level = kwargs.get("thinking_level")
             if thinking_level:
                 # In 1.56.0+, thinking_level is directly supported
                 thinking_config = types.ThinkingConfig(
                     include_thoughts=False,
                     thinking_level=thinking_level
                 )

        return types.GenerateContentConfig(
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            max_output_tokens=kwargs.get("max_tokens", 8192),
            thinking_config=thinking_config,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Generate chat completion"""
        model_name = model or self.default_model
        
        # Extract system instruction
        system_instruction = next(
            (msg.content for msg in messages if msg.role == "system"), None
        )
        
        # Apply task parameters
        self._apply_task_parameters(kwargs, model_name)

        # Convert messages to Gemini format (Content objects)
        contents = []
        for msg in messages:
            if msg.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.content)]))
            elif msg.role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg.content)]))

        config = self._get_generation_config(model_name, **kwargs)
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            response = await self.client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            if not response.candidates or not response.candidates[0].content.parts:
                raise ProviderAPIError("Gemini API returned empty response")

            return ChatResponse(
                content=response.text, usage=self._extract_usage_stats(response)
            )
        except Exception as e:
            raise ProviderAPIError(f"Gemini API error: {e}") from e

    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[dict, None]:
        """Generate streaming chat completion"""
        model_name = model or self.default_model
        
        # Extract system instruction
        system_instruction = next(
            (msg.content for msg in messages if msg.role == "system"), None
        )
        
        # Apply task parameters
        self._apply_task_parameters(kwargs, model_name)

        contents = []
        for msg in messages:
            if msg.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.content)]))
            elif msg.role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg.content)]))

        config = self._get_generation_config(model_name, **kwargs)
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )

            async for chunk in response_stream:
                if chunk:
                    try:
                        if chunk.candidates:
                            candidate = chunk.candidates[0]
                            if candidate.content and candidate.content.parts:
                                text = candidate.content.parts[0].text
                                if text:
                                    yield {"text": text}
                            
                            # Check finish reason
                            if candidate.finish_reason and candidate.finish_reason != "STOP":
                                yield {"error": f"Gemini generation stopped. Finish reason: {candidate.finish_reason}"}

                        usage = self._extract_usage_stats(chunk)
                        if usage:
                            yield {"usage": usage}
                    except Exception as e:
                        yield {"error": f"Gemini streaming chunk processing error: {e}"}
        except Exception as e:
            yield {"error": f"Gemini streaming API error: {e}"}

    @property
    def supports_caching(self) -> bool:
        return True  # Gemini supports context caching

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_async_batch(self) -> bool:
        return True