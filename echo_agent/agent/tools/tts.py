"""Text-to-speech tool — convert text to audio via edge-tts or OpenAI TTS."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from echo_agent.agent.tools.base import Tool, ToolExecutionContext, ToolResult


class TTSTool(Tool):
    name = "text_to_speech"
    description = "Convert text to speech audio. Uses edge-tts (free) by default, or OpenAI TTS if configured."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to convert to speech."},
            "voice": {"type": "string", "description": "Voice name. For edge-tts: e.g., 'en-US-AriaNeural'. For OpenAI: 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'."},
            "output_path": {"type": "string", "description": "Output file path (relative to workspace). Auto-generated if omitted."},
            "backend": {"type": "string", "enum": ["edge", "openai"], "description": "TTS backend to use."},
        },
        "required": ["text"],
    }
    timeout_seconds = 60

    def __init__(self, workspace: str, openai_api_key: str = ""):
        self._workspace = Path(workspace)
        self._openai_key = openai_api_key

    async def execute(self, params: dict[str, Any], ctx: ToolExecutionContext | None = None) -> ToolResult:
        text = params["text"]
        backend = params.get("backend", "edge" if not self._openai_key else "openai")
        voice = params.get("voice", "")
        output_path = params.get("output_path", "")

        if not output_path:
            output_path = f"tts_output_{id(text) % 100000}.mp3"
        full_path = (self._workspace / output_path).resolve()

        if backend == "openai":
            return await self._openai_tts(text, voice or "alloy", full_path)
        return await self._edge_tts(text, voice or "en-US-AriaNeural", full_path)

    async def _edge_tts(self, text: str, voice: str, output: Path) -> ToolResult:
        try:
            import edge_tts
        except ImportError:
            return ToolResult(success=False, error="edge-tts not installed: pip install edge-tts")

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output))
            return ToolResult(output=f"Audio saved to {output.name}", metadata={"path": str(output), "voice": voice})
        except Exception as e:
            return ToolResult(success=False, error=f"edge-tts failed: {e}")

    async def _openai_tts(self, text: str, voice: str, output: Path) -> ToolResult:
        if not self._openai_key:
            return ToolResult(success=False, error="OpenAI API key not configured for TTS")

        import aiohttp
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {self._openai_key}", "Content-Type": "application/json"}
        body = {"model": "tts-1", "input": text, "voice": voice, "response_format": "mp3"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        return ToolResult(success=False, error=f"OpenAI TTS error {resp.status}: {err[:300]}")
                    data = await resp.read()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(data)
                    return ToolResult(output=f"Audio saved to {output.name}", metadata={"path": str(output), "voice": voice})
        except Exception as e:
            return ToolResult(success=False, error=f"OpenAI TTS failed: {e}")
