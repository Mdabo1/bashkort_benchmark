"""Сборка тестовых наборов из открытых источников с зафиксированными ревизиями.

Каждая функция пишет testsets/<name>/data.jsonl. Что можно распространять
(CC BY / CDLA), коммитится; FLORES (CC BY-SA) и аудио скачиваются локально.
"""

import collections
import random
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

from ..io import CACHE, TESTSETS, read_jsonl, write_json, write_jsonl
from ..normalize import BASHKIR_SPECIFIC, normalize_for_wer

HF = "https://huggingface.co"

SOURCES = {
    "flores200": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    "phrasebooks": ("failed09/bashkir-multilingual-phrasebooks", "eae6488f6adb8ff582b7089c9ffb3a6261cb3cc4",
                    "cleaned.parquet"),
    "periodicals": ("bashkorttele/periodicals-izddom", "c18e45e9d76de8cfce2609c82c609988dd36ca66",
                    "data/data.jsonl"),
    "aigiz_parallel": ("AigizK/bashkir-russian-parallel-corpora", "0cddd5ffe7fffa8e23fd64a94a52429668513bbc",
                       "data/train-00000-of-00001.parquet"),
    "broadcast": ("bashkorttele/broadcast-speech", "5d9dc895c2adf24257b750f99864b69036d509f0", None),
}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"↓ {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def hf_file(repo: str, revision: str, path: str) -> Path:
    url = f"{HF}/datasets/{repo}/resolve/{revision}/{path}"
    return download(url, CACHE / "hf" / repo / revision / path)


# --- перевод -----------------------------------------------------------------

def build_flores() -> None:
    tgz = download(SOURCES["flores200"], CACHE / "flores200_dataset.tar.gz")
    out = CACHE / "flores200"
    if not out.exists():
        with tarfile.open(tgz) as t:
            t.extractall(out, filter="data")
    base = out / "flores200_dataset"
    for split in ("dev", "devtest"):
        ba = (base / split / f"bak_Cyrl.{split}").read_text(encoding="utf-8").splitlines()
        ru = (base / split / f"rus_Cyrl.{split}").read_text(encoding="utf-8").splitlines()
        assert len(ba) == len(ru)
        rows = [{"id": f"flores-{split}-{i:04d}", "ba": b.strip(), "ru": r.strip()}
                for i, (b, r) in enumerate(zip(ba, ru))]
        write_jsonl(TESTSETS / f"flores200-{split}" / "data.jsonl", rows)
        print(f"flores200-{split}: {len(rows)}")


def build_phrasebook() -> None:
    """Разговорные фразы из разговорников БГПУ (через исправленную версию failed09).

    Берём только записи типа phrase, без вариантов в скобках и многоточий,
    не короче двух слов, без повторов после нормализации.
    """
    import pyarrow.parquet as pq

    repo, rev, path = SOURCES["phrasebooks"]
    rows = pq.read_table(hf_file(repo, rev, path)).to_pylist()
    seen, out = set(), []
    for r in rows:
        if r["entry_type"] != "phrase":
            continue
        ba, ru = r["text_ba"].strip(), r["text_ru"].strip()
        if re.search(r"[()\[\]/]|\.\.\.|…", ba + ru):
            continue
        if len(ba.split()) < 2 or len(ru.split()) < 1:
            continue
        key = normalize_for_wer(ba)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": r["id"], "ba": ba, "ru": ru, "source_book": r["source_book"]})
    write_jsonl(TESTSETS / "phrasebook-v1" / "data.jsonl", out)
    print(f"phrasebook-v1: {len(out)}")


def contamination(testset: str) -> dict:
    """Сколько башкирских сторон набора дословно (после нормализации) есть в корпусе Айгиза.

    Корпус на 1,2 млн пар — главный открытый источник для обучения ba↔ru,
    поэтому совпадения с ним помечаются, а метрики считаются и без них.
    """
    import pyarrow.parquet as pq

    repo, rev, path = SOURCES["aigiz_parallel"]
    table = pq.read_table(hf_file(repo, rev, path), columns=["ba", "ru"])
    corpus_ba = {normalize_for_wer(s) for s in table.column("ba").to_pylist() if s}
    corpus_ru = {normalize_for_wer(s) for s in table.column("ru").to_pylist() if s}
    p = TESTSETS / testset / "data.jsonl"
    rows = read_jsonl(p)
    n_ba = n_ru = 0
    for r in rows:
        r["in_aigiz_corpus_ba"] = normalize_for_wer(r["ba"]) in corpus_ba
        r["in_aigiz_corpus_ru"] = normalize_for_wer(r["ru"]) in corpus_ru
        n_ba += r["in_aigiz_corpus_ba"]
        n_ru += r["in_aigiz_corpus_ru"]
    write_jsonl(p, rows)
    report = {"testset": testset, "n": len(rows), "ba_exact_in_corpus": n_ba, "ru_exact_in_corpus": n_ru,
              "corpus": f"{repo}@{rev}"}
    write_json(TESTSETS / testset / "contamination.json", report)
    print(report)
    return report


# --- грамотность: минимальные пары -------------------------------------------

# Башкирская буква → то, что пишут на её месте без башкирской раскладки.
STRIP_LETTERS = {"ҡ": "к", "ғ": "г", "ҙ": "з", "ҫ": "с", "һ": "х", "ң": "н", "ә": "э", "ө": "о", "ү": "у"}

# Пары окончаний, различающихся только гармонией гласных (заднерядный ↔ переднерядный).
# Список нарочно короткий: только частые аффиксы, где замена почти наверняка даёт ошибку.
HARMONY_SUFFIXES = [
    ("ларҙы", "ләрҙе"), ("тарҙы", "тәрҙе"), ("дарҙы", "дәрҙе"), ("ҙарҙы", "ҙәрҙе"),
    ("лары", "ләре"), ("тары", "тәре"), ("дары", "дәре"), ("ҙары", "ҙәре"),
    ("ларға", "ләргә"), ("тарға", "тәргә"), ("дарға", "дәргә"), ("ҙарға", "ҙәргә"),
    ("лар", "ләр"), ("тар", "тәр"), ("дар", "дәр"), ("ҙар", "ҙәр"), ("нар", "нәр"),
    ("ның", "нең"), ("тың", "тең"), ("дың", "дең"), ("ҙың", "ҙең"),
    ("нан", "нән"), ("тан", "тән"), ("дан", "дән"), ("ҙан", "ҙән"),
    ("ға", "гә"), ("ҡа", "кә"),
    ("ныҡ", "нек"),
]

def _strip_letter(ch: str) -> str:
    low = ch.lower()
    if low not in STRIP_LETTERS:
        return ch
    return STRIP_LETTERS[low].upper() if ch.isupper() else STRIP_LETTERS[low]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁӘӨҮҠҒҘҪҺҢ«“\"])")
_TOKEN = re.compile(r"[а-яёәөүҡғҙҫһң]+(?:-[а-яёәөүҡғҙҫһң]+)*", re.IGNORECASE)


def _iter_periodicals(limit_records: int | None = None):
    repo, rev, path = SOURCES["periodicals"]
    import json

    with open(hf_file(repo, rev, path), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit_records and i >= limit_records:
                break
            r = json.loads(line)
            if r.get("text_source") != "native":
                continue
            yield r


def _sentences(text: str):
    text = unicodedata.normalize("NFC", text)
    for s in _SENT_SPLIT.split(text):
        s = " ".join(s.split())
        # « - » с пробелами — обычно след переноса из вёрстки («бәй - ләнеш»), такие предложения пропускаем
        if 6 <= len(s.split()) <= 25 and s[-1] in ".!?" and not re.search(r"[A-Za-z0-9]| - ", s):
            yield s


def build_orth_pairs(n_per_kind: int = 500, seed: int = 7) -> None:
    """Пары «правильное предложение / то же с одной ошибкой».

    Предложения — из периодики bashkorttele (CC BY 4.0), только записи с родным
    текстом (не OCR). Частоты слов считаются по всей периодике: исходное слово
    должно встречаться часто (≥ 20 раз), испорченное — почти никогда (≤ 1 раз).
    Это страхует от случаев, когда «ошибка» на самом деле другое правильное слово.
    """
    freq = collections.Counter()
    records = []
    for r in _iter_periodicals():
        toks = [t.lower() for t in _TOKEN.findall(r["text"])]
        freq.update(toks)
        records.append(r)
    lex_path = CACHE / "lexicon" / "periodicals_freq.tsv"
    lex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lex_path, "w", encoding="utf-8") as f:
        for w, c in freq.most_common():
            if c >= 3:
                f.write(f"{w}\t{c}\n")

    rng = random.Random(seed)
    rng.shuffle(records)
    # Тестовые предложения берём из первых 2000 записей после перемешивания,
    # чтобы один материал не давал много пар.
    pool = []
    for r in records[:2000]:
        sents = list(_sentences(r["text"]))
        rng.shuffle(sents)
        pool += [(r["id"], s) for s in sents[:3]]
    rng.shuffle(pool)

    def corrupt_letters(sent):
        toks = [m for m in _TOKEN.finditer(sent) if set(m.group(0).lower()) & BASHKIR_SPECIFIC]
        rng.shuffle(toks)
        for m in toks:
            w = m.group(0)
            if freq[w.lower()] < 20 or (w.isupper() and len(w) > 1):
                continue
            bad = "".join(_strip_letter(ch) for ch in w)
            if bad.lower() == w.lower() or freq[bad.lower()] > 1:
                continue
            return sent[:m.start()] + bad + sent[m.end():], w, bad
        return None

    def corrupt_harmony(sent):
        toks = list(_TOKEN.finditer(sent))
        rng.shuffle(toks)
        for m in toks:
            w = m.group(0)
            lw = w.lower()
            # только слова в нижнем регистре: заглавные буквы — это имена и заголовки
            if freq[lw] < 20 or "-" in w or w != lw:
                continue
            for back, front in HARMONY_SUFFIXES:
                for a, b in ((back, front), (front, back)):
                    if lw.endswith(a) and len(lw) > len(a) + 1:
                        bad = w[: len(w) - len(a)] + b
                        if freq[bad.lower()] > 1:
                            continue
                        return sent[:m.start()] + bad + sent[m.end():], w, bad
        return None

    out = []
    for kind, fn in (("letters", corrupt_letters), ("harmony", corrupt_harmony)):
        k = 0
        for src_id, s in pool:
            if k >= n_per_kind:
                break
            res = fn(s)
            if not res:
                continue
            bad_sent, w, bad = res
            out.append({"id": f"orth-{kind}-{k:04d}", "kind": kind, "good": s, "bad": bad_sent,
                        "word_good": w, "word_bad": bad, "source_id": src_id,
                        "freq_good": freq[w.lower()], "freq_bad": freq[bad.lower()], "verified": False})
            k += 1
        # одно предложение — одна пара на тип
        used = {o["good"] for o in out}
        pool = [p for p in pool if p[1] not in used]
    write_jsonl(TESTSETS / "orth-pairs-v1" / "data.jsonl", out)
    print(f"orth-pairs-v1: {collections.Counter(o['kind'] for o in out)}")


BUILDERS = {
    "flores200-devtest": build_flores,
    "flores200-dev": build_flores,
    "phrasebook-v1": build_phrasebook,
    "orth-pairs-v1": build_orth_pairs,
}
