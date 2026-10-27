"""Нормализация башкирского текста перед подсчётом WER/CER и для орфографических метрик.

Правило одно для всех систем: регистр, пунктуация и латинские двойники букв
не должны влиять на рейтинг, а настоящие орфографические ошибки (к вместо ҡ,
қ вместо ҡ) должны.
"""

import re
import unicodedata

# Башкирский алфавит: 33 буквы русского + 9 своих.
BASHKIR_SPECIFIC = set("әөүҡғҙҫһң")
RUSSIAN = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
BASHKIR_ALPHABET = RUSSIAN | BASHKIR_SPECIFIC

# Буквы соседних кириллических алфавитов, которых нет в башкирском.
# Их появление обычно значит, что модель пишет по-казахски/по-татарски
# или путает похожие буквы (қ ↔ ҡ, ҙ ↔ ӡ).
FOREIGN_CYRILLIC = {
    "қ": "kk",  # казахская ка (в башкирском ҡ)
    "ұ": "kk",
    "і": "kk",
    "җ": "tt",
    "ӣ": "tg",
    "ӯ": "tg",
    "ҷ": "tg",
    "ӡ": "other",
    "ӊ": "other",
    "ҝ": "other",
    "ӑ": "cv",
    "ӗ": "cv",
}

# Латинские двойники, которые встречаются в текстах, набранных без башкирской
# раскладки. Заменяются только внутри слов, где есть кириллица.
LATIN_HOMOGLYPHS = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "k": "к", "m": "м", "t": "т", "b": "в", "h": "һ", "ə": "ә", "ç": "ҫ",
    "ö": "ө", "ü": "ү", "ğ": "ғ", "ñ": "ң",
}

_CYR = re.compile(r"[Ѐ-ӿ]")
_WORD = re.compile(r"[\ẁ-ͯ]+", re.UNICODE)
_DASHES = re.compile(r"[-‐‑‒–—―]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def fix_homoglyphs(text: str) -> str:
    """Заменяет латинские двойники кириллическими в словах со смешанным письмом."""

    def repl(m: re.Match) -> str:
        w = m.group(0)
        if not _CYR.search(w):
            return w
        return "".join(LATIN_HOMOGLYPHS.get(ch, ch) for ch in w)

    return _WORD.sub(repl, text)


def normalize_for_wer(text: str) -> str:
    """Нижний регистр, без пунктуации, дефисы и тире — пробелы, Unicode NFC.

    Цифры не раскрываются в слова: это известное ограничение, оно одинаково
    для всех систем и описано в docs/methodology.md.
    """
    t = unicodedata.normalize("NFC", text).lower()
    t = fix_homoglyphs(t)
    t = _DASHES.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    t = t.replace("_", " ")
    return _SPACES.sub(" ", t).strip()


def words(text: str) -> list[str]:
    return normalize_for_wer(text).split()
