"""Google Imagen provider using the new google-genai SDK"""

import mimetypes
import time
import httpx
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from google import genai
from google.genai import types
from .base import BaseProvider, Message, ChatResponse, ProviderAPIError
from ..settings import TaskType, IMAGE_GENERATION_CONFIG


class ImagenProvider(BaseProvider):
    """Google Imagen provider"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        enable_caching: bool = True,
        default_model: Optional[str] = None,
    ):
        super().__init__(api_key, enable_caching, default_model)
        self.base_url = base_url
        
        http_options = {}
        if base_url:
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
                
            http_options['base_url'] = clean_base_url
            if api_version:
                http_options['api_version'] = api_version

        # Force httpx usage to avoid "Chunk too big" error from aiohttp
        http_options['async_client_args'] = {'transport': httpx.AsyncHTTPTransport()}

        # Initialize the new Google Gen AI client
        self.client = genai.Client(
            api_key=api_key,
            http_options=http_options if http_options else None
        )

    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate images using Imagen models"""
        model_name = model or self.default_model

        # Extract prompt from the last user message
        prompt = next((msg.content for msg in reversed(messages) if msg.role == "user"), None)
        
        if not prompt:
            raise ProviderAPIError("No prompt provided for image generation")
            
        config_args = {}
        # Allow kwargs to override defaults, otherwise use global config
        config_args["image_size"] = kwargs.get("image_size", IMAGE_GENERATION_CONFIG["image_size"])
        config_args["aspect_ratio"] = kwargs.get("aspect_ratio", IMAGE_GENERATION_CONFIG["aspect_ratio"])
        
        # Add other optional parameters if present in kwargs
        for key in ["number_of_images", "person_generation", "safety_filter_level"]:
            if key in kwargs:
                config_args[key] = kwargs[key]
                
        config = types.GenerateImagesConfig(**config_args)
        
        try:
            response = await self.client.aio.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=config,
            )
            
            full_text = ""
            file_index = 0
            output_dir = kwargs.get("output_dir", Path("."))
            if isinstance(output_dir, str):
                output_dir = Path(output_dir)
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            if response.generated_images:
                for generated_image in response.generated_images:
                    if generated_image.image:
                        # Mimic GeminiProvider saving logic
                        # Note: GeneratedImage.image is usually bytes or has bytes
                        image_data = generated_image.image
                        
                        # Handle if it's an Image object from the SDK
                        if hasattr(image_data, "image_bytes"):
                             image_bytes = image_data.image_bytes
                        else:
                             image_bytes = image_data
                             
                        file_extension = ".png" # Imagen defaults to png
                        file_name = f"image_{int(time.time())}_{file_index}{file_extension}"
                        file_path = output_dir / file_name
                        
                        with open(file_path, "wb") as f:
                            f.write(image_bytes)

                        full_text += f"\n[Image saved to {file_path}]\n"
                        file_index += 1
            else:
                # Attempt to extract detailed failure reason from response
                reason = "No images were generated by the model."
                
                # Check for common safety/filter fields in the response object
                # Some versions of the SDK provide feedback or filters
                feedback = getattr(response, "prompt_feedback", None)
                if feedback:
                    reason += f" Prompt feedback: {feedback}"
                
                filters = getattr(response, "filters", None)
                if filters:
                    reason += f" Safety filters triggered: {filters}"
                
                raise ProviderAPIError(f"Imagen API returned empty result: {reason}")

            return ChatResponse(
                content=full_text, usage=None
            )
        except Exception as e:
            # Extract as much detail as possible from the exception
            error_details = str(e)
            
            # Check for enhanced error info if available on the exception object
            if hasattr(e, "message") and e.message:
                error_details = e.message
            
            # Add context for common HTTP/API error codes
            if "400" in error_details:
                error_details += " (The request was invalid. Check if your prompt violates safety guidelines or contains unsupported characters.)"
            elif "401" in error_details or "403" in error_details:
                error_details += " (Authentication failed. Please verify your API key and ensure the Imagen API is enabled for your project.)"
            elif "429" in error_details:
                error_details += " (Rate limit exceeded. Please wait before sending more requests.)"
            elif "500" in error_details or "503" in error_details:
                error_details += " (The server encountered an error or is overloaded. Try again later.)"
                
            raise ProviderAPIError(f"Imagen API request failed: {error_details}") from e

    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[dict, None]:
        """Generate streaming chat completion - Delegates to chat_completion for images"""
        # For image generation tasks, use non-streaming mode as it's more robust
        response = await self.chat_completion(messages, model=model, **kwargs)
        yield {"text": response.content}
        if response.usage:
            yield {"usage": response.usage}

    @property
    def supports_caching(self) -> bool:
        return False  # Imagen typically doesn't use context caching in the same way

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_async_batch(self) -> bool:
        return True
