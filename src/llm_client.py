"""Unified LLM client for DeepSeek and Kimi (OpenAI-compatible APIs)."""
import os
import time
from typing import Optional

import yaml


class LLMClient:
    """Simple OpenAI-compatible API client.

    Reads config from ``config.yaml``; API key from environment or ``.env``.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.provider = cfg['llm']['provider']
        self.model = cfg['llm']['model']
        self.base_url = cfg['llm']['base_url']
        self.api_key = os.getenv(cfg['llm']['api_key_env'])
        self.temperature = cfg['llm'].get('temperature', 0.1)
        self.max_tokens = cfg['llm'].get('max_tokens', 4096)
        self.retry = cfg['llm'].get('retry', 3)
        self.backoff = cfg['llm'].get('backoff', 2.0)

        if self.api_key is None:
            raise ValueError(
                f"API key not found in env var {cfg['llm']['api_key_env']}"
            )

        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, system_prompt: str, user_message: str, history: Optional[list] = None) -> str:
        """Send a chat request and return assistant content.

        Args:
            system_prompt: System-level instruction.
            user_message: Current user turn.
            history: Previous messages as list of dicts {'role': ..., 'content': ...}.

        Returns:
            Assistant response text.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if history:
            messages.extend(history)
        messages.append({'role': 'user', 'content': user_message})

        for attempt in range(1, self.retry + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content
                if content is None:
                    return ""
                return content.strip()
            except Exception as e:
                if attempt == self.retry:
                    raise RuntimeError(f"LLM API failed after {self.retry} retries: {e}")
                time.sleep(self.backoff * attempt)
