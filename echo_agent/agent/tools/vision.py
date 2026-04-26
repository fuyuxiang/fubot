"""Vision tool — analyze images via a vision-capable LLM."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolPermission, ToolResult
from echo_agent.models.provider import LLMProvider


def _encode_image(path: str) -> tuple[str, str]:
    ext = Path(path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    data = Path(path).read_bytes()
    return mime, base64.b64encode(data).decode()


class VisionTool(Tool):
    name = "vision_analyze"
    description = "Analyze an image using a vision-capable LLM. Provide a file path or URL and a question about the image."
    parameters = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Path to a local image file or an image URL."},
            "prompt": {"type": "string", "description": "Question or instruction about the image."},
            "model": {"type": "string", "description": "Override model for vision. Uses the configured default if omitted."},
        },
        "required": ["image", "prompt"],
    }
    required_permissions = [ToolPermission.READ, ToolPermission.NETWORK]
    timeout_seconds = 60

    def __init__(self, provider: LLMProvider, workspace: str = ""):
        self._provider = provider
        self._workspace = workspace

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        image = params["image"]
        prompt = params["prompt"]
        model = params.get("model")

        if image.startswith(("http://", "https://")):
            image_content = {"type": "image_url", "image_url": {"url": image}}
        else:
            image_path = Path(image)
            if not image_path.is_absolute() and self._workspace:
                image_path = Path(self._workspace) / image_path
            if not image_path.exists():
                return ToolResult(success=False, error=f"Image not found: {image}")
            try:
                mime, b64 = _encode_image(str(image_path))
                image_content = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to read image: {e}")

        messages = [
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": prompt},
            ]},
        ]

        resp = await self._provider.chat_with_retry(messages=messages, model=model)
        if resp.finish_reason == "error":
            return ToolResult(success=False, error=resp.content or "Vision analysis failed")
        return ToolResult(output=resp.content or "(no analysis)")

    def execution_mode(self, params: dict[str, Any]) -> str:
        return "read_only"
