"""Наборы для распознавания речи.

cv25-test           — чтение по подсказке (Common Voice 25, тестовая часть). Скачивается
                      вручную с Mozilla Data Collective, путь — в CV25_BA_DIR.
broadcast-spont-v1  — живая речь из теле- и радиоэфира (bashkorttele/broadcast-speech).
                      Фрагменты выбраны VAD-ом без участия какой-либо ASR-системы,
                      эталоны пишут носители по human_eval/GUIDELINES_ASR.md.
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import CACHE, TESTSETS, read_jsonl, write_json, write_jsonl
from .build import SOURCES, hf_file

SEED = 2026
PER_PROGRAMME = 2        # записей на программу
SEG_PER_RECORDING = 7    # фрагментов на запись
MIN_S, MAX_S = 4.0, 20.0
SKIP_EDGES_S = 45.0      # заставки и музыка в начале и конце


def build_cv25() -> None:
    root = os.environ.get("CV25_BA_DIR")
    if not root:
        raise SystemExit("Укажите CV25_BA_DIR — папку Common Voice 25 (ba) с test.tsv и clips/")
    root = Path(root)
    rows = []
    with open(root / "test.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"id": r["path"].rsplit(".", 1)[0], "audio": str((root / "clips" / r["path"]).resolve()),
                         "text": r["sentence"], "meta": {"client_id": r["client_id"][:12],
                                                         "gender": r.get("gender", ""), "age": r.get("age", "")}})
    write_jsonl(TESTSETS / "cv25-test" / "data.jsonl", rows)
    print(f"cv25-test: {len(rows)} (данные не коммитятся: пути локальные)")


def _segments(wav: np.ndarray, vad) -> list[tuple[float, float]]:
    segs = next(iter(vad.segment_batch(wav[None, :], np.array([len(wav)]), 16000,
                                       min_silence_duration_ms=300, max_speech_duration_s=MAX_S,
                                       speech_pad_ms=150)))
    return [(s / 16000, e / 16000) for s, e in segs]


def build_broadcast(per_programme: int = PER_PROGRAMME, seg_per_rec: int = SEG_PER_RECORDING) -> None:
    import onnx_asr

    from ..systems.asr import load_audio

    repo, rev, _ = SOURCES["broadcast"]
    meta = list(csv.DictReader(open(hf_file(repo, rev, "metadata.csv"), encoding="utf-8")))
    by_prog = defaultdict(list)
    for m in meta:
        if m.get("source_quality", "normal") != "normal":
            continue
        by_prog[m["programme_slug"]].append(m)

    rng = random.Random(SEED)
    chosen = []
    for prog in sorted(by_prog):
        recs = sorted(by_prog[prog], key=lambda m: m["file_name"])
        rng.shuffle(recs)
        chosen += recs[:per_programme]
    print(f"записей: {len(chosen)}, программ: {len(by_prog)}")

    vad = onnx_asr.load_vad("silero")
    rows = []
    for m in chosen:
        local = hf_file(repo, rev, m["file_name"])
        wav = load_audio(str(local))
        dur = len(wav) / 16000
        segs = [(s, e) for s, e in _segments(wav, vad)
                if MIN_S <= e - s <= MAX_S and s > min(SKIP_EDGES_S, dur * 0.1) and e < dur - min(SKIP_EDGES_S, dur * 0.1)]
        if not segs:
            continue
        # равномерно по записи: делим на seg_per_rec отрезков и берём по случайному фрагменту из каждого
        bins = defaultdict(list)
        for s, e in segs:
            bins[int(s / dur * seg_per_rec)].append((s, e))
        picked = [rng.choice(bins[b]) for b in sorted(bins)][:seg_per_rec]
        rel = str(Path("hf") / repo / rev / m["file_name"])
        for s, e in picked:
            rows.append({
                "id": f"{Path(m['file_name']).parent.name}-{Path(m['file_name']).stem}-{int(s * 1000):08d}",
                "audio": rel, "start": round(s, 2), "end": round(e, 2), "text": "",
                "meta": {"programme": m["programme_slug"], "programme_ba": m.get("programme_ba", ""),
                         "medium": m.get("medium", ""), "organization": m.get("organization", ""),
                         "title_ba": m.get("title_ba", "")},
            })
        print(f"  {m['file_name']}: {len(picked)} фрагм.")

    out = TESTSETS / "broadcast-spont-v1" / "data.jsonl"
    # Уже введённые эталоны не теряем при пересборке
    if out.exists():
        old = {r["id"]: r for r in read_jsonl(out)}
        for r in rows:
            if r["id"] in old and old[r["id"]].get("text"):
                r["text"] = old[r["id"]]["text"]
                r["annotation"] = old[r["id"]].get("annotation")
    write_jsonl(out, rows)
    total = sum(r["end"] - r["start"] for r in rows)
    write_json(TESTSETS / "broadcast-spont-v1" / "stats.json", {
        "segments": len(rows), "minutes": round(total / 60, 1), "recordings": len(chosen),
        "programmes": len({r["meta"]["programme"] for r in rows}), "source": f"{repo}@{rev}", "seed": SEED})
    print(f"broadcast-spont-v1: {len(rows)} фрагментов, {total / 60:.1f} мин")


def export_clips(testset: str = "broadcast-spont-v1", out_dir: str | None = None) -> Path:
    """Нарезает фрагменты в отдельные файлы для разметчиков (16 кГц, моно, FLAC)."""
    import soundfile as sf

    from ..systems.asr import load_audio

    out = Path(out_dir or CACHE / "clips" / testset)
    out.mkdir(parents=True, exist_ok=True)
    for r in read_jsonl(TESTSETS / testset / "data.jsonl"):
        p = out / f"{r['id']}.flac"
        if not p.exists():
            sf.write(p, load_audio(str(CACHE / r["audio"]), r["start"], r["end"]), 16000)
    return out


def fetch_broadcast(force_reselect: bool = False) -> None:
    """Список фрагментов в репозитории — источник истины: к нему привязаны расшифровки.
    Обычная сборка только докачивает нужные записи. Заново выбирать фрагменты
    (BASHBENCH_RESELECT=1) нужно лишь при выпуске новой версии набора."""
    out = TESTSETS / "broadcast-spont-v1" / "data.jsonl"
    if force_reselect or os.environ.get("BASHBENCH_RESELECT") or not out.exists():
        return build_broadcast()
    repo, rev, _ = SOURCES["broadcast"]
    prefix = f"hf/{repo}/{rev}/"
    files = sorted({r["audio"] for r in read_jsonl(out)})
    for rel in files:
        assert rel.startswith(prefix), rel
        hf_file(repo, rev, rel[len(prefix):])
    print(f"broadcast-spont-v1: аудио на месте ({len(files)} записей)")


BUILDERS = {"cv25-test": build_cv25, "broadcast-spont-v1": fetch_broadcast}
