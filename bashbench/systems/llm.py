"""Языковые модели: API с OpenAI-совместимым протоколом, ГигаЧат и локальные HF-модели."""

import json
import ssl
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from .base import LLMSystem, env, retry


class OpenAICompatible(LLMSystem):
    """Любой сервер с /chat/completions: YandexGPT (OpenAI-совместимый эндпоинт),
    vLLM, OpenRouter и т. п. Ключ — из переменной `key_env`."""

    def __init__(self, name: str, model: str, base_url: str, key_env: str | None = None,
                 extra_headers: dict | None = None, temperature: float = 0.0, workers: int = 4):
        self.name, self.model, self.base_url = name, model, base_url.rstrip("/")
        self.key_env, self.extra_headers = key_env, extra_headers or {}
        self.temperature, self.workers = temperature, workers

    def _headers(self):
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self.key_env:
            h["Authorization"] = f"Bearer {env(self.key_env)}"
        return h

    def _one(self, prompt, system, max_tokens):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = json.dumps({"model": self.model, "messages": msgs, "temperature": self.temperature,
                           "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=self._headers())
        resp = retry(lambda: json.load(urllib.request.urlopen(req, timeout=120)))
        return resp["choices"][0]["message"]["content"] or ""

    def chat(self, prompts, system=None, max_tokens=512):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self._one(p, system, max_tokens), prompts))

    def info(self):
        import re

        # ID каталога Yandex Cloud в результатах не публикуем
        model = re.sub(r"^gpt://[^/]+/", "gpt://<folder>/", self.model)
        return {"name": self.name, "model": model, "base_url": self.base_url, "temperature": self.temperature}


class GigaChat(OpenAICompatible):
    """ГигаЧат (Сбер). Нужен GIGACHAT_AUTH_KEY — ключ авторизации (Basic) из личного кабинета.

    Сертификат НУЦ Минцифры: если Python ему не доверяет, укажите путь к нему
    в GIGACHAT_CA_BUNDLE.
    """

    OAUTH = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def __init__(self, model: str = "GigaChat-2-Max", scope: str = "GIGACHAT_API_PERS", **kw):
        kw.setdefault("workers", 1)  # бесплатный Freemium-режим — один поток
        super().__init__(name=model.lower(), model=model, base_url="https://gigachat.devices.sberbank.ru/api/v1", **kw)
        self.scope = scope
        self._token, self._exp = None, 0.0

    def _ctx(self):
        import os

        ca = os.environ.get("GIGACHAT_CA_BUNDLE")
        return ssl.create_default_context(cafile=ca) if ca else None

    def _headers(self):
        import os

        # Готовый токен (живёт 30 минут) — когда OAuth-сервер на порту 9443 недоступен из сети,
        # где идёт прогон: токен получают на другой машине и передают в GIGACHAT_ACCESS_TOKEN.
        if os.environ.get("GIGACHAT_ACCESS_TOKEN"):
            return {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['GIGACHAT_ACCESS_TOKEN']}"}
        if time.time() > self._exp - 60:
            req = urllib.request.Request(self.OAUTH, data=urllib.parse.urlencode({"scope": self.scope}).encode(),
                                         headers={"Authorization": f"Basic {env('GIGACHAT_AUTH_KEY')}",
                                                  "RqUID": str(uuid.uuid4()),
                                                  "Content-Type": "application/x-www-form-urlencoded"})
            resp = json.load(urllib.request.urlopen(req, timeout=30, context=self._ctx()))
            self._token, self._exp = resp["access_token"], resp["expires_at"] / 1000
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._token}"}

    def _one(self, prompt, system, max_tokens):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = json.dumps({"model": self.model, "messages": msgs, "temperature": self.temperature or 0.01,
                           "max_tokens": max_tokens}).encode()

        def call():
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=self._headers())
            return json.load(urllib.request.urlopen(req, timeout=120, context=self._ctx()))

        return retry(call)["choices"][0]["message"]["content"] or ""


def yandexgpt(model: str = "yandexgpt/latest") -> OpenAICompatible:
    """YandexGPT через OpenAI-совместимый эндпоинт Yandex Cloud.
    Нужны YANDEX_API_KEY и YANDEX_FOLDER_ID."""
    import os

    folder = os.environ.get("YANDEX_FOLDER_ID", "<folder>")
    sys_ = OpenAICompatible(name=model.split("/")[0], model=f"gpt://{folder}/{model}",
                            base_url="https://llm.api.cloud.yandex.net/v1", key_env="YANDEX_API_KEY")
    return sys_


class HFCausalLM(LLMSystem):
    """Локальная модель с открытыми весами. Даёт и генерацию, и log-вероятности —
    для минимальных пар это честнее, чем спрашивать модель «какой вариант верный»."""

    has_logprobs = True

    def __init__(self, model_id: str, chat_template: bool | None = None, batch: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id.split("/")[-1]
        self.model_id, self.batch, self.torch = model_id, batch, torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        except TypeError:
            # Goldfish хранит конфиг токенизатора ALBERT, который transformers 5 не разбирает;
            # tokenizer.json при этом рабочий. Goldfish ждут [CLS] в начале текста.
            from huggingface_hub import hf_hub_download
            from transformers import PreTrainedTokenizerFast

            self.tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(model_id, "tokenizer.json"),
                                               bos_token="[CLS]", eos_token="[SEP]", pad_token="<pad>")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.use_chat = chat_template if chat_template is not None else bool(getattr(self.tok, "chat_template", None))

    def logprob(self, texts):
        out = []
        bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        for s in range(0, len(texts), self.batch):
            chunk = texts[s:s + self.batch]
            ids = [[bos] + self.tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
            L = max(map(len, ids))
            pad = self.tok.pad_token_id
            inp = self.torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=self.device)
            mask = self.torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=self.device)
            with self.torch.inference_mode():
                logits = self.model(input_ids=inp, attention_mask=mask).logits.float()
            lp = self.torch.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:, None])[..., 0]
            lp = (lp * mask[:, 1:]).sum(-1)
            out += lp.tolist()
        return out

    def chat(self, prompts, system=None, max_tokens=512):
        outs = []
        for p in prompts:
            if self.use_chat:
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                text = p
            enc = self.tok(text, return_tensors="pt", add_special_tokens=not self.use_chat).to(self.device)
            with self.torch.inference_mode():
                g = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
            outs.append(self.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    def info(self):
        return {"name": self.name, "model": self.model_id, "device": self.device}
