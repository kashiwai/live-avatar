import os
from dataclasses import dataclass
from typing import Generator, List, Dict, Any

from openai import OpenAI


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.6


class OpenAIChatAgent:
    def __init__(self, cfg: OpenAIConfig):
        if not cfg.api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key)

    def chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

