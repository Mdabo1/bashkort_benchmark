"""Прогон системы на тестовом наборе. Результат дописывается порциями,
так что прерванный прогон (лимиты API, обрыв сети) продолжается с места остановки."""

import datetime
import hashlib
import json
import random
import subprocess

from .io import ROOT, load_testset, read_jsonl, run_path, testset_path, write_json

CHUNK = 64

# Что поддерживает каждый набор: дорожка и допустимые задачи.
TESTSETS = {
    "flores200-devtest": ("mt", ["ba-ru", "ru-ba"]),
    "flores200-dev": ("mt", ["ba-ru", "ru-ba"]),
    "phrasebook-v1": ("mt", ["ba-ru", "ru-ba"]),
    "orth-pairs-v1": ("llm", ["choice"]),
    "gen-v1": ("llm", ["gen"]),
    "cv25-test": ("asr", ["asr"]),
    "broadcast-spont-v1": ("asr", ["asr"]),
}

CHOICE_PROMPT = (
    "Ниже два варианта одного предложения на башкирском языке. Один написан правильно, "
    "во втором есть орфографическая или грамматическая ошибка.\n\nA: {a}\nB: {b}\n\n"
    "Какой вариант правильный? Ответь одной буквой: A или B."
)

GEN_SYSTEM = "Ты отвечаешь только на литературном башкирском языке (кириллица, башкирская орфография)."


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _parse_choice(text: str) -> str | None:
    """Первое отдельно стоящее A/B (латиница или кириллица): «B», «**A**.», «Вариант А»."""
    import re

    m = re.search(r"(?<![\w])([ABАВ])(?![\w])", text.upper())
    return None if not m else {"А": "A", "В": "B"}.get(m.group(1), m.group(1))


def run(testset: str, task: str, system_name: str, limit: int | None = None) -> None:
    track, tasks = TESTSETS[testset]
    if task not in tasks:
        raise ValueError(f"{testset}: задачи {tasks}, а не {task}")
    items = load_testset(testset)[:limit] if limit else load_testset(testset)
    out_path = run_path(track, testset, task, system_name)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{system_name} → {testset}/{task}: {len(todo)} из {len(items)} осталось")
    if not todo:
        return

    from . import systems

    src = tgt = None
    if track == "mt":
        sys_ = systems.mt_system(system_name)
        src, tgt = task.split("-")
    elif track == "asr":
        sys_ = systems.asr_system(system_name)
    else:
        sys_ = systems.llm_system(system_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"system": sys_.info(), "testset": testset, "task": task, "testset_sha": _sha(testset_path(testset)),
            "bashbench_rev": _git_rev(), "started": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}

    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), CHUNK):
            chunk = todo[s:s + CHUNK]
            rows = _process(track, task, sys_, chunk, src, tgt)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  записано {len(done) + s + len(chunk)}/{len(items)}")

    meta["finished"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    write_json(out_path.with_suffix(".meta.json"), meta)


def _process(track, task, sys_, chunk, src, tgt):
    if track == "mt":
        hyps = sys_.translate([it[src] for it in chunk], src, tgt)
        return [{"id": it["id"], "hyp": h} for it, h in zip(chunk, hyps)]

    if track == "asr":
        from .systems.asr import load_audio

        audio = [load_audio(_audio_path(it), it.get("start"), it.get("end")) for it in chunk]
        hyps = sys_.transcribe(audio)
        return [{"id": it["id"], "hyp": h} for it, h in zip(chunk, hyps)]

    if task == "choice":
        if sys_.has_logprobs:
            lp = sys_.logprob([it["good"] for it in chunk] + [it["bad"] for it in chunk])
            n = len(chunk)
            return [{"id": it["id"], "method": "logprob", "lp_good": g, "lp_bad": b, "correct": g > b}
                    for it, g, b in zip(chunk, lp[:n], lp[n:])]
        rows, prompts = [], []
        for it in chunk:
            # позиция правильного варианта детерминированно случайна для каждой пары
            good_first = random.Random(it["id"]).random() < 0.5
            a, b = (it["good"], it["bad"]) if good_first else (it["bad"], it["good"])
            prompts.append(CHOICE_PROMPT.format(a=a, b=b))
            rows.append((it, "A" if good_first else "B"))
        outs = sys_.chat(prompts, max_tokens=5)
        return [{"id": it["id"], "method": "prompt", "answer": o, "pred": _parse_choice(o), "gold": gold,
                 "correct": _parse_choice(o) == gold} for (it, gold), o in zip(rows, outs)]

    if task == "gen":
        outs = sys_.chat([it["prompt"] for it in chunk], system=GEN_SYSTEM, max_tokens=600)
        return [{"id": it["id"], "hyp": o} for it, o in zip(chunk, outs)]

    raise ValueError(task)


def _audio_path(it) -> str:
    from pathlib import Path

    from .io import CACHE

    p = Path(it["audio"])
    return str(p if p.is_absolute() else CACHE / p)
