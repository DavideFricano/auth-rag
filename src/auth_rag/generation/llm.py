from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests
from openai import OpenAI

from auth_rag.types import Message


class BaseLLMClient(ABC):
    """Common interface for LLM backends. Consumes prompts as a neutral list[Message];
    prompt assembly is the augmenter's job, not the client's."""

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        pass

    @abstractmethod
    def answer(self, messages: list[Message], **kwargs) -> str:
        pass


class OllamaClient(BaseLLMClient):
    """LLM client for locally-hosted Ollama models via the /api/chat endpoint."""

    def __init__(self, model: str, url: str = "http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    def _payload(self, messages: list[Message], temperature: float, num_ctx: int, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }

    def stream(self, messages: list[Message], temperature: float = 0.1, num_ctx: int = 4096) -> Iterator[str]:
        response = requests.post(self.url, json=self._payload(messages, temperature, num_ctx, stream=True), stream=True, timeout=800)
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done", False):
                break

    def answer(self, messages: list[Message], temperature: float = 0.1, num_ctx: int = 4096) -> str:
        response = requests.post(self.url, json=self._payload(messages, temperature, num_ctx, stream=False), timeout=800)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI chat models. Reads the API key from the environment."""

    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def stream(self, messages: list[Message], temperature: float = 0.1) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def answer(self, messages: list[Message], temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content
