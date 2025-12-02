"""Google Gemini provider"""

import asyncio
from typing import AsyncGenerator, List, Optional
import google.generativeai as genai
from google.api_core.client_options import ClientOptions
from .base import BaseProvider, Message, ChatResponse, ProviderAPIError


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
        # Extract hostname from base_url for client_options
        from urllib.parse import urlparse
        parsed_url = urlparse(base_url)
        api_endpoint = parsed_url.netloc

        genai.configure(
            api_key=api_key, client_options=ClientOptions(api_endpoint=api_endpoint)
        )
        self._model = None
        self._current_system_instruction = None

    def _get_model(
        self, model_name: Optional[str] = None, system_instruction: Optional[str] = None
    ):
        """Get or create Gemini model instance"""
        model_name = model_name or self.default_model
        if (
            self._model is None
            or self._model.model_name != model_name
            or self._current_system_instruction != system_instruction
        ):
            self._model = genai.GenerativeModel(
                model_name, system_instruction=system_instruction
            )
            self._current_system_instruction = system_instruction
        return self._model

    async def chat_completion(
        self, messages: List[Message], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Generate chat completion"""
        # Extract system instruction
        system_instruction = next(
            (msg.content for msg in messages if msg.role == "system"), None
        )
        model_instance = self._get_model(model, system_instruction)

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})
            # Skip system messages as Gemini handles them differently (passed to model init)

        try:
            # Use context caching if available and enabled
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
            }

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            if len(gemini_messages) == 1:
                # Single message
                response = await model_instance.generate_content_async(
                    gemini_messages[0]["parts"][0],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            else:
                # Multi-turn conversation
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
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
        # Extract system instruction
        system_instruction = next(
            (msg.content for msg in messages if msg.role == "system"), None
        )
        model_instance = self._get_model(model, system_instruction)

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

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            if len(gemini_messages) == 1:
                response_stream = await model_instance.generate_content_async(
                    gemini_messages[0]["parts"][0],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
                )
            else:
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                response_stream = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
                )

            iterator = aiter(response_stream)
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=30.0)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield {"error": "Error: Gemini streaming timed out. Please try '/mode sync' or check your network connection."}
                    break
                except Exception as e:
                    yield {"error": f"Gemini streaming chunk error: {e}"}
                    break

                if chunk:
                    try:
                        # Check for blocking feedback
                        if hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback:
                            # In newer SDKs block_reason might be an enum, checking existence
                            pass 

                        if chunk.candidates:
                            candidate = chunk.candidates[0]
                            if candidate.content and candidate.content.parts:
                                text = candidate.content.parts[0].text
                                if text:
                                    yield {"text": text}
                            
                            # Check finish reason if text is empty/missing but candidate exists
                            if candidate.finish_reason and candidate.finish_reason != 0: # 0 is STOP usually
                                # If blocked by safety, it might show up here
                                pass

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
