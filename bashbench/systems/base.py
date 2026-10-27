"""Общий интерфейс систем. Каждая система — класс с методом под свою дорожку.

Закрытые API читают ключи из переменных окружения и падают с понятной
ошибкой, если ключа нет. Никакие ключи в результатах не сохраняются.
"""

import os
import time


class MissingCredentials(RuntimeError):
    pass


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise MissingCredentials(f"Нужна переменная окружения {name}")
    return v


def retry(fn, tries: int = 5, base: float = 2.0):
    for i in range(tries):
        try:
            return fn()
        except MissingCredentials:
            raise
        except Exception as e:  # сеть, лимиты API
            if i == tries - 1:
                raise
            wait = base ** (i + 1)
            print(f"  повтор через {wait:.0f} с: {e}")
            time.sleep(wait)


class MTSystem:
    name: str

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name}


class ASRSystem:
    name: str

    def transcribe(self, audio: list) -> list[str]:
        """audio — список моно-сигналов 16 кГц (numpy float32)."""
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name}


class LLMSystem:
    """Языковая модель: генерация и (для открытых весов) оценка вероятности текста."""

    name: str
    has_logprobs = False

    def chat(self, prompts: list[str], system: str | None = None, max_tokens: int = 512) -> list[str]:
        raise NotImplementedError

    def logprob(self, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name}
