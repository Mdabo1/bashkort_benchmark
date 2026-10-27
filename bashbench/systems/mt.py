"""Системы перевода ba↔ru."""

import json
import urllib.request

from .base import MTSystem, env, retry

FLORES_CODES = {"ba": "bak_Cyrl", "ru": "rus_Cyrl"}
LANG_NAMES_RU = {"ba": "башкирский", "ru": "русский"}


class NLLB(MTSystem):
    """Открытая NLLB-200 (Meta) через transformers. Работает на CPU, на GPU быстрее."""

    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", beams: int = 4, batch: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = model_id.split("/")[-1]
        self.model_id = model_id
        self.beams, self.batch = beams, batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device).eval()
        self.torch = torch

    def translate(self, texts, src, tgt):
        self.tok.src_lang = FLORES_CODES[src]
        bos = self.tok.convert_tokens_to_ids(FLORES_CODES[tgt])
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=self.beams,
                                          max_new_tokens=int(enc["input_ids"].shape[1] * 1.6) + 10)
            for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t
            print(f"  {min(s + self.batch, len(texts))}/{len(texts)}", end="\r")
        print()
        return out

    def info(self):
        return {"name": self.name, "model": self.model_id, "beams": self.beams, "device": self.device}


class NLLBCT2(MTSystem):
    """NLLB через CTranslate2 с int8-весами: на CPU в разы быстрее transformers.
    Модель конвертируется при первом запуске в .cache/ct2/."""

    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", beams: int = 4, batch: int = 32):
        import os
        import subprocess

        import ctranslate2
        from transformers import AutoTokenizer

        from ..io import CACHE

        self.model_id, self.beams, self.batch = model_id, beams, batch
        self.name = model_id.split("/")[-1] + "-ct2int8"
        path = CACHE / "ct2" / f"{model_id.split('/')[-1]}-int8"
        if not (path / "model.bin").exists():
            subprocess.run(["ct2-transformers-converter", "--model", model_id, "--output_dir", str(path),
                            "--quantization", "int8", "--force"], check=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tr = ctranslate2.Translator(str(path), device="cpu", compute_type="int8",
                                         intra_threads=os.cpu_count() or 4)

    def translate(self, texts, src, tgt):
        self.tok.src_lang = FLORES_CODES[src]
        toks = [self.tok.convert_ids_to_tokens(self.tok.encode(t)) for t in texts]
        res = self.tr.translate_batch(toks, target_prefix=[[FLORES_CODES[tgt]]] * len(toks), beam_size=self.beams,
                                      max_batch_size=self.batch, max_decoding_length=512)
        out = []
        for r in res:
            ids = self.tok.convert_tokens_to_ids(r.hypotheses[0][1:])
            out.append(self.tok.decode(ids, skip_special_tokens=True))
        return out

    def info(self):
        return {"name": self.name, "model": self.model_id, "beams": self.beams, "runtime": "ctranslate2 int8"}


class YandexTranslate(MTSystem):
    """Yandex Cloud Translate API v2. Нужны YANDEX_API_KEY и YANDEX_FOLDER_ID."""

    name = "yandex-translate"
    URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    def translate(self, texts, src, tgt):
        key, folder = env("YANDEX_API_KEY"), env("YANDEX_FOLDER_ID")
        out = []
        for s in range(0, len(texts), 50):  # API берёт до 10 000 символов за запрос
            chunk = texts[s:s + 50]
            body = json.dumps({"folderId": folder, "texts": chunk, "sourceLanguageCode": src,
                               "targetLanguageCode": tgt}).encode()
            req = urllib.request.Request(self.URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Api-Key {key}"})
            resp = retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))
            out += [t["text"] for t in resp["translations"]]
        return out


class GoogleTranslate(MTSystem):
    """Google Cloud Translation v2 (Basic). Нужен GOOGLE_API_KEY."""

    name = "google-translate"
    URL = "https://translation.googleapis.com/language/translate/v2"

    def translate(self, texts, src, tgt):
        import html

        key = env("GOOGLE_API_KEY")
        out = []
        for s in range(0, len(texts), 100):
            chunk = texts[s:s + 100]
            body = json.dumps({"q": chunk, "source": src, "target": tgt, "format": "text"}).encode()
            req = urllib.request.Request(f"{self.URL}?key={key}", data=body,
                                         headers={"Content-Type": "application/json"})
            resp = retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))
            out += [html.unescape(t["translatedText"]) for t in resp["data"]["translations"]]
        return out


MT_PROMPT = ("Переведи текст с языка «{src}» на язык «{tgt}». Выведи только перевод, "
             "без пояснений, кавычек и транслитерации.\n\nТекст: {text}")


class LLMTranslator(MTSystem):
    """Любая LLM из systems/llm.py как переводчик (ГигаЧат, YandexGPT, открытые модели)."""

    def __init__(self, llm):
        self.llm = llm
        self.name = f"{llm.name}"

    def translate(self, texts, src, tgt):
        prompts = [MT_PROMPT.format(src=LANG_NAMES_RU[src], tgt=LANG_NAMES_RU[tgt], text=t) for t in texts]
        return [o.strip().strip("«»\"").split("\n")[0].strip() for o in self.llm.chat(prompts, max_tokens=400)]

    def info(self):
        return {**self.llm.info(), "prompt": MT_PROMPT}
