"""Google Gemini provider"""

from typing import AsyncGenerator, List, Optional
import google.generativeai as genai
from .base import BaseProvider, Message, ChatResponse


class GeminiProvider(BaseProvider):
    """Google Gemini provider"""

    def __init__(
        self,
        api_key: str,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
    ):
        super().__init__(api_key, enable_caching, default_model)
        genai.configure(api_key=api_key)
        self._model = None
        self._sync_client = genai.GenerativeModel(self.default_model)

    def _get_model(self, model_name: Optional[str] = None):
        """Get or create Gemini model instance"""
        model_name = model_name or self.default_model
        if self._model is None or self._model.model_name != model_name:
            self._model = genai.GenerativeModel(model_name)
        return self._model

    def chat_completion_sync(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Generate sync chat completion"""
        model_name = model or self.default_model
        model_instance = genai.GenerativeModel(model_name)

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})

        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )

            if len(gemini_messages) == 1:
                response = model_instance.generate_content(
                    gemini_messages[0]["parts"][0], generation_config=generation_config
                )
            else:
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(
                    gemini_messages[-1]["parts"][0], generation_config=generation_config
                )

            return ChatResponse(
                content=response.text, usage=self._extract_usage_stats(response)
            )
        except Exception as e:
            raise RuntimeError(f"Gemini sync API error: {e}")

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Generate chat completion"""
        model_instance = self._get_model(model)

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})
            # Skip system messages as Gemini handles them differently

        try:
            # Use context caching if available and enabled
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
            }

            if len(gemini_messages) == 1:
                # Single message
                response = await model_instance.generate_content_async(
                    gemini_messages[0]["parts"][0], generation_config=generation_config
                )
            else:
                # Multi-turn conversation
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0], generation_config=generation_config
                )

            if not response.candidates or not response.candidates[0].content.parts:
                raise RuntimeError("Gemini API returned empty response")
            
            return ChatResponse(
                content=response.text, 
                usage=self._extract_usage_stats(response)
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    async def stream_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion"""
        model_instance = self._get_model(model)

        gemini_messages = []
        for msg in messages:
            if msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})

        try:
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
            }

            if len(gemini_messages) == 1:
                # Await the async generator creation
                response_stream = await model_instance.generate_content_async(
                    gemini_messages[0]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )
            else:
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                # Await the async generator creation
                response_stream = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )

            async for chunk in response_stream:
                if chunk:
                    try:
                        if chunk.candidates and chunk.candidates[0].content.parts:
                            text = chunk.candidates[0].content.parts[0].text
                            if text:
                                yield {
                                    "text": text.strip(),
                                    "usage": self._extract_usage_stats(chunk)
                                }
                    except Exception as e:
                        yield {
                            "error": f"\n[Stream Error: {str(e)}]",
                            "usage": None
                        }
        except Exception as e:
            raise RuntimeError(f"Gemini streaming error: {e}")

    @property
    def supports_caching(self) -> bool:
        return True  # Gemini supports context caching

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_async_batch(self) -> bool:
        return True
