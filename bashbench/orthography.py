"""Дешёвые автоматические сигналы для текста, который LLM пишет «по-башкирски».

Они не заменяют оценку носителей (LLM-судьи и такие метрики плохо ловят
грамматику), но сразу показывают грубые провалы: ответ по-русски или
по-татарски, казахские буквы, слова, которых нет в корпусе.
"""

import re
import unicodedata
from functools import lru_cache

from .io import CACHE
from .normalize import BASHKIR_ALPHABET, BASHKIR_SPECIFIC, FOREIGN_CYRILLIC

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_CYR = re.compile(r"^[Ѐ-ӿ-]+$")
LID_MODEL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


@lru_cache(1)
def _lid():
    import fasttext

    from .data.build import download

    path = download(LID_MODEL, CACHE / "lid.176.ftz")
    fasttext.FastText.eprint = lambda *a, **k: None
    return fasttext.load_model(str(path))


@lru_cache(1)
def lexicon() -> set[str]:
    p = CACHE / "lexicon" / "periodicals_freq.tsv"
    if not p.exists():
        raise FileNotFoundError("Нет частотного словаря: python -m bashbench build orth-pairs-v1")
    return {line.split("\t", 1)[0] for line in p.open(encoding="utf-8")}


def language(text: str) -> tuple[str, float]:
    t = " ".join(text.split())
    if not t:
        return "none", 0.0
    # FastText.predict ломается на numpy 2 (np.array(copy=False)), поэтому зовём нижний уровень
    (prob, label), = _lid().f.predict(t + "\n", 1, 0.0, "strict")
    return label.replace("__label__", ""), float(prob)


def analyze(text: str) -> dict:
    t = unicodedata.normalize("NFC", text).lower()
    ws = _WORD.findall(t)
    cyr = [w for w in ws if _CYR.match(w)]
    n = max(len(cyr), 1)
    lex = lexicon()
    lang, p = language(text)
    return {
        "lang": lang,
        "lang_p": p,
        "n_words": len(ws),
        "latin_words": sum(1 for w in ws if re.search(r"[a-z]", w)) / max(len(ws), 1),
        "bashkir_letter_words": sum(1 for w in cyr if set(w) & BASHKIR_SPECIFIC) / n,
        "foreign_letter_words": sum(1 for w in cyr if set(w) & set(FOREIGN_CYRILLIC)) / n,
        "non_alphabet_words": sum(1 for w in cyr if set(w.replace("-", "")) - BASHKIR_ALPHABET) / n,
        "oov_words": sum(1 for w in cyr if w not in lex and not all(p in lex for p in w.split("-"))) / n,
    }
