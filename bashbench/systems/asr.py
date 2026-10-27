"""Системы распознавания башкирской речи."""

import io
import json
import os
import time
import urllib.request
import uuid

import numpy as np

from .base import ASRSystem, env, retry

SR = 16000


def load_audio(path: str, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Моно 16 кГц float32. start/end — секунды, для фрагментов длинных записей."""
    import soundfile as sf

    info = sf.info(path)
    s = int((start or 0) * info.samplerate)
    e = int(end * info.samplerate) if end is not None else -1
    wav, sr = sf.read(path, start=s, stop=None if e < 0 else e, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def wav_bytes(wav: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class GigaAMBashkirONNX(ASRSystem):
    """AigizK/GigaAM-Bashkir-CV25-ONNX через onnx-asr (CPU, INT8 по умолчанию)."""

    def __init__(self, quantization: str | None = "int8"):
        import onnx_asr

        self.quantization = quantization
        self.name = "gigaam-bashkir-cv25" + (f"-{quantization}" if quantization else "")
        self.model = onnx_asr.load_model("AigizK/GigaAM-Bashkir-CV25-ONNX", quantization=quantization)

    def transcribe(self, audio):
        out = []
        for i in range(0, len(audio), 16):
            out += list(self.model.recognize(audio[i:i + 16], sample_rate=SR))
            print(f"  {min(i + 16, len(audio))}/{len(audio)}", end="\r")
        print()
        return out

    def info(self):
        return {"name": self.name, "model": "AigizK/GigaAM-Bashkir-CV25-ONNX", "quantization": self.quantization}


class HFCTC(ASRSystem):
    """CTC-модели из transformers: MMS (target_lang=bak), w2v-bert/wav2vec2 Айгиза и др."""

    def __init__(self, model_id: str, target_lang: str | None = None, name: str | None = None):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_id, self.target_lang, self.torch = model_id, target_lang, torch
        self.name = name or model_id.split("/")[-1]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"target_lang": target_lang, "ignore_mismatched_sizes": True} if target_lang else {}
        self.proc = AutoProcessor.from_pretrained(model_id, **({"target_lang": target_lang} if target_lang else {}))
        self.model = AutoModelForCTC.from_pretrained(model_id, **kw).to(self.device).eval()
        if target_lang:
            self.proc.tokenizer.set_target_lang(target_lang)
            self.model.load_adapter(target_lang)

    def transcribe(self, audio):
        out = []
        for n, wav in enumerate(audio):
            enc = self.proc(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**enc).logits
            out.append(self.proc.batch_decode(logits.argmax(-1))[0])
            print(f"  {n + 1}/{len(audio)}", end="\r")
        print()
        return out

    def info(self):
        return {"name": self.name, "model": self.model_id, "target_lang": self.target_lang}


class EmbeddingGemma2Bashkir(ASRSystem):
    """AigizK/EmbeddingGemma2-Bashkir-CV25. Загрузчик (bashkir_gemma.py) лежит в репозитории модели
    и выполняется как код — так модель распространяет автор."""

    name = "embeddinggemma2-bashkir-cv25"

    def __init__(self):
        import sys

        import torch
        from huggingface_hub import snapshot_download

        d = snapshot_download("AigizK/EmbeddingGemma2-Bashkir-CV25")
        sys.path.insert(0, d)
        from bashkir_gemma import BashkirGemma

        self.asr = BashkirGemma.from_pretrained(d, device="cuda" if torch.cuda.is_available() else "cpu")

    def transcribe(self, audio):
        import tempfile

        import soundfile as sf

        out = []
        with tempfile.TemporaryDirectory() as tmp:
            for n, w in enumerate(audio):
                p = f"{tmp}/{n}.wav"
                sf.write(p, w, SR)
                r = self.asr.transcribe(p)
                out.append(r if isinstance(r, str) else getattr(r, "text", str(r)))
        return out


class HFWhisper(ASRSystem):
    def __init__(self, model_id: str = "AigizK/bashkir-whisper-small", language: str | None = None):
        from transformers import pipeline

        self.model_id, self.language = model_id, language
        self.name = model_id.split("/")[-1]
        self.pipe = pipeline("automatic-speech-recognition", model=model_id)

    def transcribe(self, audio):
        kw = {"generate_kwargs": {"language": self.language, "task": "transcribe"}} if self.language else {}
        return [self.pipe({"raw": w, "sampling_rate": SR}, **kw)["text"] for w in audio]


class OmnilingualASR(ASRSystem):
    """Meta Omnilingual ASR. Ставится отдельно: pip install omnilingual-asr (тянет fairseq2)."""

    def __init__(self, model_card: str = "omniASR_LLM_7B", lang: str = "bak_Cyrl"):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.model_card, self.lang = model_card, lang
        self.name = model_card.lower().replace("_", "-")
        self.pipe = ASRInferencePipeline(model_card=model_card)

    def transcribe(self, audio):
        inp = [{"waveform": w, "sample_rate": SR} for w in audio]
        return self.pipe.transcribe(inp, lang=[self.lang] * len(inp), batch_size=4)

    def info(self):
        return {"name": self.name, "model_card": self.model_card, "lang": self.lang}


class YandexSpeechKit(ASRSystem):
    """SpeechKit, синхронное распознавание (API v1, до 30 с на запрос).

    Нужны YANDEX_API_KEY, YANDEX_FOLDER_ID. Код языка — YANDEX_STT_LANG
    (по умолчанию ba-RU; проверьте актуальный список языков в документации).
    """

    name = "yandex-speechkit"
    URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"

    def transcribe(self, audio):
        key, folder = env("YANDEX_API_KEY"), env("YANDEX_FOLDER_ID")
        lang = os.environ.get("YANDEX_STT_LANG", "ba-RU")
        out = []
        for w in audio:
            pcm = (np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()
            url = f"{self.URL}?lang={lang}&folderId={folder}&format=lpcm&sampleRateHertz={SR}"
            req = urllib.request.Request(url, data=pcm, headers={"Authorization": f"Api-Key {key}"})
            out.append(retry(lambda: json.load(urllib.request.urlopen(req, timeout=60))).get("result", ""))
        return out

    def info(self):
        return {"name": self.name, "lang": os.environ.get("YANDEX_STT_LANG", "ba-RU")}


class Speechmatics(ASRSystem):
    """Speechmatics Batch API, language=ba. Нужен SPEECHMATICS_API_KEY."""

    name = "speechmatics"
    URL = "https://asr.api.speechmatics.com/v2/jobs"

    def _req(self, url, data=None, headers=None):
        h = {"Authorization": f"Bearer {env('SPEECHMATICS_API_KEY')}", **(headers or {})}
        return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=120)

    def _submit(self, wav):
        boundary = uuid.uuid4().hex
        cfg = json.dumps({"type": "transcription", "transcription_config": {"language": "ba"}})
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"config\"\r\n\r\n{cfg}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"data_file\"; filename=\"a.wav\"\r\n"
                f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes(wav) + f"\r\n--{boundary}--\r\n".encode()
        r = self._req(self.URL, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return json.load(r)["id"]

    def transcribe(self, audio):
        jobs = [retry(lambda w=w: self._submit(w)) for w in audio]
        out = []
        for j in jobs:
            while True:
                st = json.load(self._req(f"{self.URL}/{j}"))["job"]["status"]
                if st in ("done", "rejected"):
                    break
                time.sleep(3)
            out.append(self._req(f"{self.URL}/{j}/transcript?format=txt").read().decode() if st == "done" else "")
        return out
