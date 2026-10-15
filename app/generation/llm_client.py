"""Async Ollama client with streaming, health check, and graceful offline fallback."""

import json
from typing import AsyncGenerator, Optional
import httpx

from app.core.exceptions import OllamaConnectionError
from app.core.logger import get_logger

logger = get_logger("llm_client")


class OllamaClient:
    """Async client communicating with local Ollama daemon."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    async def check_health(self) -> bool:
        """Verifies whether the local Ollama instance is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def generate(self, prompt: str) -> str:
        """Sends a complete prompt to Ollama and returns the synthesized text response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                else:
                    logger.warning(
                        f"Ollama returned HTTP {response.status_code}: {response.text}"
                    )
                    return self._fallback_synthesis(prompt)

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                f"Could not connect to Ollama at {self.base_url} ({exc}). Using offline synthesis."
            )
            return self._fallback_synthesis(prompt)
        except Exception as exc:
            logger.error(f"Unexpected error querying Ollama: {exc}", exc_info=True)
            return self._fallback_synthesis(prompt)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streams generation tokens from Ollama as Server-Sent Events (SSE)."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': 'Ollama generation failed'})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk_json = json.loads(line)
                                token = chunk_json.get("response", "")
                                is_done = chunk_json.get("done", False)

                                yield f"data: {json.dumps({'token': token, 'done': is_done})}\n\n"
                                if is_done:
                                    break
                            except Exception:
                                continue

        except Exception as exc:
            logger.warning(f"Ollama streaming failed ({exc}). Emitting fallback stream.")
            fallback_text = self._fallback_synthesis(prompt)
            yield f"data: {json.dumps({'token': fallback_text, 'done': True})}\n\n"

    def _fallback_synthesis(self, prompt: str) -> str:
        """Deterministic offline fallback when Ollama daemon is offline or starting up."""
        return (
            "[Ollama Offline Mode] Retrieved relevant context from document chunks above. "
            "Please start the Ollama daemon (`ollama serve`) and pull the model (`ollama run llama3.2:3b`) "
            "for complete neural generation."
        )
