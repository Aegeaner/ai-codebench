"""Tencent Hunyuan AIART provider"""

import asyncio
import json
import os
import sys
import time
import mimetypes
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any

import httpx
import aiofiles
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.aiart.v20221229 import aiart_client, models

from .base import BaseProvider, Message, ChatResponse, ProviderAPIError


class HunyuanProvider(BaseProvider):
    """Tencent Hunyuan AIART provider"""

    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        region: str = "ap-shanghai",
        base_url: str = "aiart.eu-frankfurt.tencentcloudapi.com",
        enable_caching: bool = True,
        default_model: Optional[str] = None,
    ):
        # BaseProvider expects a single api_key, but we need two.
        # We'll pass the secret_id as api_key to satisfy the parent,
        # but store both correctly.
        super().__init__(secret_id, enable_caching, default_model)
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.base_url = base_url

        cred = credential.Credential(self.secret_id, self.secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = self.base_url

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        
        self.client = aiart_client.AiartClient(cred, self.region, clientProfile)

    async def _submit_job(self, prompt: str, **kwargs) -> str:
        """Submit text-to-image job and return JobId"""
        def _submit():
            req = models.SubmitTextToImageJobRequest()
            params = {
                "Prompt": prompt,
                "LogoAdd": 0,  # No watermark
            }
            
            # Apply resolution if provided, else let API default or use config
            # Resolution format "1024:1024"
            if "resolution" in kwargs:
                params["Resolution"] = kwargs["resolution"]
            
            req.from_json_string(json.dumps(params))
            resp = self.client.SubmitTextToImageJob(req)
            return resp.JobId

        return await asyncio.to_thread(_submit)

    async def _query_job(self, job_id: str) -> Dict[str, Any]:
        """Query job status"""
        def _query():
            req = models.QueryTextToImageJobRequest()
            params = {"JobId": job_id}
            req.from_json_string(json.dumps(params))
            resp = self.client.QueryTextToImageJob(req)
            return {
                "status": resp.JobStatusMsg,
                "status_code": str(resp.JobStatusCode), # "1": Waiting, "2": Running, "4": Failed, "5": Success
                "result_image": resp.ResultImage,
                "error_msg": resp.JobErrorMsg
            }

        return await asyncio.to_thread(_query)

    async def _download_image(self, url: str, output_dir: Path) -> str:
        """Download image from URL and save to file"""
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.content
                
                # Guess extension or default to png (Hunyuan usually returns jpg/png)
                content_type = response.headers.get("content-type")
                ext = mimetypes.guess_extension(content_type) or ".jpg"
                
                filename = f"hunyuan_{int(time.time())}_{os.urandom(4).hex()}{ext}"
                file_path = output_dir / filename
                
                # Use asyncio.to_thread for blocking mkdir
                await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
                
                # Use aiofiles for non-blocking file write
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(data)
                
                return str(file_path)
            else:
                raise ProviderAPIError(f"Failed to download image: {response.status_code}")

    def _handle_tencent_error(self, e: TencentCloudSDKException) -> str:
        """Map Tencent Cloud error codes to user-friendly messages"""
        code = e.code if hasattr(e, "code") else ""
        
        error_map = {
            "AuthFailure.UnauthorizedOperation": "Tencent Cloud API authorization failed. Check your CAM policy.",
            "FailedOperation.GenerateImageFailed": "Image generation failed due to content moderation.",
            "FailedOperation.ImageDownloadError": "Failed to download the generated image from Tencent servers.",
            "FailedOperation.JobNotExist": "The specified job ID does not exist.",
            "FailedOperation.RequestTimeout": "Tencent Cloud service timeout.",
            "FailedOperation.ServerError": "Tencent Cloud internal server error.",
            "InvalidParameterValue.TextLengthExceed": "The prompt is too long for Hunyuan AIART.",
            "OperationDenied.ImageIllegalDetected": "The generated image was blocked by content filters.",
            "OperationDenied.TextIllegalDetected": "The prompt was blocked by content filters.",
            "RequestLimitExceeded": "Tencent Cloud API rate limit exceeded.",
            "RequestLimitExceeded.JobNumExceed": "Too many concurrent tasks. Please wait and try again.",
            "ResourceUnavailable.InArrears": "Tencent Cloud account is in arrears.",
            "ResourceUnavailable.LowBalance": "Tencent Cloud account balance is too low.",
            "ResourceUnavailable.NotExist": "Service not activated in Tencent Cloud console.",
            "ResourceUnavailable.StopUsing": "Tencent Cloud account has stopped service.",
        }
        
        friendly_msg = error_map.get(code)
        if friendly_msg:
            return f"{friendly_msg} ({code})"
        return str(e)

    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate image from text description"""
        # Remove task from kwargs if present (not needed for image generation but passed by router)
        kwargs.pop("task", None)

        # Extract the last user message as prompt
        prompt = next(
            (msg.content for msg in reversed(messages) if msg.role == "user"), None
        )
        if not prompt:
            raise ProviderAPIError("No user prompt found for image generation")

        try:
            # 1. Submit Job
            job_id = await self._submit_job(prompt, **kwargs)
            
            # 2. Poll for results
            max_retries = 60  # 60 seconds timeout approx
            status_code = "unknown"
            for _ in range(max_retries):
                result = await self._query_job(job_id)
                status_code = result["status_code"]
                
                if status_code == "5":  # Success
                    image_urls = result.get("result_image", [])
                    if not image_urls:
                         raise ProviderAPIError("Job succeeded but no image URL returned")
                    
                    output_dir = kwargs.get("output_dir", Path("."))
                    if isinstance(output_dir, str):
                        output_dir = Path(output_dir)
                        
                    saved_paths = []
                    for url in image_urls:
                        path = await self._download_image(url, output_dir)
                        saved_paths.append(path)
                    
                    content = "\n".join([f"[Image saved to {p}]" for p in saved_paths])
                    return ChatResponse(content=content, usage={"total_tokens": 0})
                    
                elif status_code == "4":  # Failed
                    raise ProviderAPIError(f"Image generation failed: {result.get('error_msg')}")
                
                await asyncio.sleep(1)
            
            raise ProviderAPIError(f"Image generation timed out. Last status code: {status_code}")

        except TencentCloudSDKException as e:
            raise ProviderAPIError(self._handle_tencent_error(e))
        except Exception as e:
            raise ProviderAPIError(f"Hunyuan provider error: {e}")

    async def stream_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[dict, None]:
        """Stream completion (Hunyuan doesn't support streaming, so we wait and yield result)"""
        try:
            response = await self.chat_completion(messages, model, **kwargs)
            yield {"text": response.content}
            if response.usage:
                yield {"usage": response.usage}
        except Exception as e:
            print(f"Hunyuan Stream Error: {e}", file=sys.stderr)
            yield {"error": str(e)}

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_caching(self) -> bool:
        return False

    @property
    def supports_async_batch(self) -> bool:
        return False
