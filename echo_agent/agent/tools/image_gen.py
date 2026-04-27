"""Image generation tool — generate images via API."""

from __future__ import annotations

from typing import Any

import aiohttp

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult


class ImageGenTool(Tool):
    name = "image_generate"
    description = "Generate an image from a text prompt using an image generation API (OpenAI DALL-E or compatible)."
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the image to generate."},
            "size": {"type": "string", "enum": ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"], "description": "Image size."},
            "quality": {"type": "string", "enum": ["standard", "hd"], "description": "Image quality."},
        },
        "required": ["prompt"],
    }
    timeout_seconds = 120

    def __init__(self, api_key: str = "", api_base: str = "", model: str = "dall-e-3"):
        self._api_key = api_key
        self._api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self._model = model

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        if not self._api_key:
            return ToolResult(success=False, error="Image generation API key not configured")

        prompt = params["prompt"]
        size = params.get("size", "1024x1024")
        quality = params.get("quality", "standard")

        url = f"{self._api_base}/images/generations"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        body = {"model": self._model, "prompt": prompt, "size": size, "quality": quality, "n": 1}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return ToolResult(success=False, error=f"API error {resp.status}: {text[:500]}")
                    data = await resp.json()
                    image_url = data["data"][0].get("url", "")
                    revised = data["data"][0].get("revised_prompt", "")
                    output = f"Image URL: {image_url}"
                    if revised:
                        output += f"\nRevised prompt: {revised}"
                    return ToolResult(output=output, metadata={"url": image_url})
        except Exception as e:
            return ToolResult(success=False, error=f"Image generation failed: {e}")
