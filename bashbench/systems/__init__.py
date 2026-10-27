"""Реестр систем: имя → фабрика. Тяжёлые зависимости импортируются лениво."""


def _llm(name):
    from . import llm

    table = {
        "gigachat-2-max": lambda: llm.GigaChat("GigaChat-2-Max"),
        "gigachat-2-pro": lambda: llm.GigaChat("GigaChat-2-Pro"),
        "yandexgpt": lambda: llm.yandexgpt("yandexgpt/latest"),
        "yandexgpt-lite": lambda: llm.yandexgpt("yandexgpt-lite/latest"),
        "goldfish-bak": lambda: llm.HFCausalLM("goldfish-models/bak_cyrl_full", chat_template=False),
    }
    if name in table:
        return table[name]()
    if name.startswith("hf:"):
        return llm.HFCausalLM(name[3:])
    if name.startswith("openai:"):
        # openai:<base_url>|<model>|<KEY_ENV>, например openai:https://openrouter.ai/api/v1|google/gemini-2.5-pro|OPENROUTER_API_KEY
        base, model, key = name[7:].split("|")
        return llm.OpenAICompatible(name=model.replace("/", "_"), model=model, base_url=base, key_env=key)
    raise KeyError(f"Неизвестная LLM: {name}")


def mt_system(name):
    from . import mt

    table = {
        "nllb-200-distilled-600M": lambda: mt.NLLB("facebook/nllb-200-distilled-600M"),
        "nllb-200-distilled-600M-ct2int8": lambda: mt.NLLBCT2("facebook/nllb-200-distilled-600M"),
        "nllb-200-distilled-1.3B": lambda: mt.NLLB("facebook/nllb-200-distilled-1.3B"),
        "nllb-200-distilled-1.3B-ct2int8": lambda: mt.NLLBCT2("facebook/nllb-200-distilled-1.3B"),
        "nllb-200-3.3B": lambda: mt.NLLB("facebook/nllb-200-3.3B", batch=4),
        "yandex-translate": mt.YandexTranslate,
        "google-translate": mt.GoogleTranslate,
    }
    if name in table:
        return table[name]()
    if name.startswith("nllb:"):
        return mt.NLLB(name[5:])
    return mt.LLMTranslator(_llm(name))


def asr_system(name):
    from . import asr

    table = {
        "gigaam-bashkir-cv25-int8": lambda: asr.GigaAMBashkirONNX("int8"),
        "gigaam-bashkir-cv25": lambda: asr.GigaAMBashkirONNX(None),
        "mms-1b-all": lambda: asr.HFCTC("facebook/mms-1b-all", target_lang="bak", name="mms-1b-all"),
        "embeddinggemma2-bashkir-cv25": asr.EmbeddingGemma2Bashkir,
        "w2v-bert-2.0-bashkort-russian-omnivoice": lambda: asr.HFCTC("AigizK/w2v-bert-2.0-bashkort-russian-omnivoice"),
        "mhubert-147-bashkort": lambda: asr.HFCTC("AigizK/mHuBERT-147-bashkort"),
        "bashkir-whisper-small": lambda: asr.HFWhisper("AigizK/bashkir-whisper-small"),
        "whisper-medium-ba": lambda: asr.HFWhisper("AigizK/whisper-medium-ba"),
        "omniasr-llm-7b": lambda: asr.OmnilingualASR("omniASR_LLM_7B"),
        "omniasr-ctc-1b": lambda: asr.OmnilingualASR("omniASR_CTC_1B"),
        "yandex-speechkit": asr.YandexSpeechKit,
        "speechmatics": asr.Speechmatics,
    }
    if name in table:
        return table[name]()
    if name.startswith("ctc:"):
        return asr.HFCTC(name[4:])
    raise KeyError(f"Неизвестная ASR-система: {name}")


def llm_system(name):
    return _llm(name)
