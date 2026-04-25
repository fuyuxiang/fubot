"""Precise token counting — provider-specific tokenization with caching."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class TokenCounter:
    """Multi-provider token counter with instance caching.

    Uses tiktoken for OpenAI models, anthropic SDK for Anthropic,
    falls back to character-based estimation.
    """

    _instances: dict[str, "TokenCounter"] = {}

    def __init__(self, provider: str = "", model: str = ""):
        self._provider = provider.lower()
        self._model = model
        self._tokenizer: Any = None
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        if self._provider in ("openai", "openrouter"):
            try:
                import tiktoken
                try:
                    self._tokenizer = tiktoken.encoding_for_model(self._model)
                except KeyError:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.debug("Using tiktoken for {}/{}", self._provider, self._model)
                return
            except ImportError:
                pass
        if self._provider in ("anthropic", "bedrock"):
            try:
                from anthropic import Anthropic
                self._tokenizer = Anthropic()
                logger.debug("Using anthropic SDK tokenizer for {}", self._model)
                return
            except ImportError:
                pass
        logger.debug("Using fallback tokenizer for {}/{}", self._provider, self._model)

    @classmethod
    def for_model(cls, provider: str, model: str) -> "TokenCounter":
        key = f"{provider}:{model}"
        if key not in cls._instances:
            cls._instances[key] = cls(provider, model)
        return cls._instances[key]

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._provider in ("openai", "openrouter") and self._tokenizer is not None:
            return self._count_tiktoken(text)
        if self._provider in ("anthropic", "bedrock") and self._tokenizer is not None:
            return self._count_anthropic(text)
        return self._count_fallback(text)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message overhead
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.count(block.get("text", ""))
            role = msg.get("role", "")
            total += self.count(role)
            # Tool calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    total += self.count(fn.get("name", ""))
                    args = fn.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    total += self.count(args)
        total += 2  # conversation overhead
        return total

    def count_tools(self, tools: list[dict[str, Any]]) -> int:
        total = 0
        for tool in tools:
            total += self.count(json.dumps(tool, ensure_ascii=False))
        return total

    def _count_tiktoken(self, text: str) -> int:
        try:
            return len(self._tokenizer.encode(text))
        except Exception:
            return self._count_fallback(text)

    def _count_anthropic(self, text: str) -> int:
        try:
            result = self._tokenizer.count_tokens(text, model=self._model or "claude-sonnet-4-20250514")
            return result
        except Exception:
            return self._count_fallback(text)

    @staticmethod
    def _count_fallback(text: str) -> int:
        return max(1, len(text) // 4)
