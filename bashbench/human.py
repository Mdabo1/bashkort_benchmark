"""Оценка носителями: выгрузка пакетов для разметки и сбор результатов.

Пакет — папка с manifest.json (то, что видит разметчик) и key.json (какой
системе принадлежит ответ; разметчику не показывается). Ответы перемешаны,
названия систем скрыты.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import RESULTS, TESTSETS, load_testset, read_jsonl, run_path, write_json, write_jsonl
from .run import TESTSETS as TS_INFO

SEED = 99


def export(testset: str, task: str, systems: list[str], per_system: int, out: str) -> None:
    track, _ = TS_INFO[testset]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    if track == "asr":
        from .data.asr_sets import export_clips

        items = [it for it in load_testset(testset) if not it.get("text")]
        clips = export_clips(testset)
        (out / "clips").mkdir(exist_ok=True)
        manifest = []
        for it in items:
            shutil.copy(clips / f"{it['id']}.flac", out / "clips" / f"{it['id']}.flac")
            manifest.append({"uid": it["id"], "audio": f"clips/{it['id']}.flac",
                             "duration": round(it["end"] - it["start"], 1)})
        write_json(out / "manifest.json", {"package": out.name, "kind": "transcribe", "testset": testset,
                                           "items": manifest})
        print(f"{out}: {len(manifest)} фрагментов на расшифровку")
        return

    gold = {it["id"]: it for it in load_testset(testset)}
    outs = {s: {r["id"]: r for r in read_jsonl(run_path(track, testset, task, s))} for s in systems}
    common = sorted(set.intersection(*(set(o) for o in outs.values())))
    ids = rng.sample(common, min(per_system, len(common)))
    src_key = task.split("-")[0] if track == "mt" else "prompt"
    tgt = task.split("-")[1] if track == "mt" else "ba"
    if tgt != "ba":
        print("Внимание: носители оценивают башкирский текст; для ba→ru нужны носители русского.")

    manifest, key = [], {}
    for i in ids:
        group = []
        for s in systems:
            uid = f"{len(key):05d}"
            key[uid] = {"system": s, "item_id": i}
            group.append({"uid": uid, "output": outs[s][i]["hyp"].strip()})
        rng.shuffle(group)
        manifest.append({"source": gold[i][src_key],
                         "source_lang": src_key if track == "mt" else "ru", "outputs": group})
    rng.shuffle(manifest)
    write_json(out / "manifest.json", {"package": out.name, "kind": "rate", "testset": testset, "task": task,
                                       "items": manifest})
    write_json(out / "key.json", key)
    print(f"{out}: {len(manifest)} заданий × {len(systems)} ответов. key.json разметчикам не отправлять.")


def import_files(files: list[str]) -> None:
    """Принимает JSON, выгруженные из human_eval/annotate.html (и key.json рядом с пакетом)."""
    for f in files:
        data = json.load(open(f, encoding="utf-8"))
        pkg = data["package"]
        if data["kind"] == "transcribe":
            _import_transcripts(data)
            continue
        key_path = Path(data.get("key_path") or Path(f).parent / "key.json")
        key = json.load(open(key_path, encoding="utf-8"))
        rows = []
        for uid, a in data["annotations"].items():
            if uid not in key:
                continue
            rows.append({"package": pkg, "annotator": data["annotator"], "uid": uid, **key[uid],
                         "testset": data["testset"], "task": data["task"], **a})
        dest = RESULTS / "human" / f"{pkg}__{data['annotator']}.jsonl"
        write_jsonl(dest, rows)
        print(f"{dest}: {len(rows)} оценок")
    summarize()


def _import_transcripts(data) -> None:
    from .metrics import asr_score

    p = TESTSETS / data["testset"] / "data.jsonl"
    rows = read_jsonl(p)
    n = 0
    for r in rows:
        a = data["annotations"].get(r["id"])
        if not a or a.get("skip"):
            continue
        entry = {"annotator": data["annotator"], "text": a["text"], "flags": a.get("flags", [])}
        anns = [x for x in r.get("annotations", []) if x["annotator"] != data["annotator"]] + [entry]
        r["annotations"] = anns
        # эталон — первая расшифровка; остальные нужны для согласия разметчиков
        r["text"] = anns[0]["text"]
        n += 1
    write_jsonl(p, rows)
    pairs = [(r["annotations"][0]["text"], r["annotations"][1]["text"]) for r in rows if len(r.get("annotations", [])) > 1]
    if pairs:
        s = asr_score([b for _, b in pairs], [a for a, _ in pairs])
        write_json(TESTSETS / data["testset"] / "agreement.json", {"pairs": len(pairs), "inter_annotator_WER": s.as_dict()})
        print(f"расхождение разметчиков: WER {s.value:.1f}% на {len(pairs)} фрагментах")
    print(f"{p}: добавлено {n} расшифровок")


def _boot_mean(vals, n=1000, seed=SEED):
    a = np.array(vals, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(a, size=(n, len(a))).mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def krippendorff_alpha_interval(units: dict) -> float | None:
    """units: {uid: [оценки разных разметчиков]}. Интервальная шкала."""
    data = [v for v in units.values() if len(v) >= 2]
    if not data:
        return None
    allv = [x for v in data for x in v]
    n = len(allv)
    do = sum(sum((a - b) ** 2 for i, a in enumerate(v) for j, b in enumerate(v) if i != j) / (len(v) - 1)
             for v in data) / n
    de = sum((a - b) ** 2 for a in allv for b in allv) / (n * (n - 1))
    return None if de == 0 else 1 - do / de


def summarize() -> None:
    rows = [r for p in sorted((RESULTS / "human").glob("*.jsonl")) for r in read_jsonl(p)]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    units = defaultdict(lambda: defaultdict(list))
    flags = defaultdict(lambda: defaultdict(int))
    for r in rows:
        g = f"{r['testset']}/{r['task']}"
        for crit in ("grammar", "meaning"):
            if r.get(crit):
                groups[g][r["system"]][crit].append(r[crit])
                units[(g, crit)][r["uid"]].append(r[crit])
        for fl in r.get("flags", []):
            flags[(g, r["system"])][fl] += 1
    summary = {}
    for g, by_sys in groups.items():
        summary[g] = {}
        for s, crits in by_sys.items():
            summary[g][s] = {c: dict(zip(("mean", "ci_low", "ci_high"), _boot_mean(v)), n=len(v)) for c, v in crits.items()}
            summary[g][s]["flags"] = dict(flags[(g, s)])
        summary[g]["_agreement"] = {c: krippendorff_alpha_interval(units[(g, c)]) for c in ("grammar", "meaning")}
        summary[g]["_annotators"] = sorted({r["annotator"] for r in rows if f"{r['testset']}/{r['task']}" == g})
    write_json(RESULTS / "human" / "summary.json", summary)
    print(f"{RESULTS / 'human' / 'summary.json'}")


def leaderboard_section(path: Path) -> list[str]:
    summary = json.load(open(path, encoding="utf-8"))
    L = ["## Оценка носителями", "",
         "Шкала 1–5. «Грамотность» — насколько текст похож на написанный грамотным носителем; "
         "«Смысл» — передан ли смысл источника (перевод) или выполнено ли задание (генерация). "
         "α — согласие разметчиков по Криппендорфу (1 — полное, 0 — случайное).", ""]
    for g, by_sys in summary.items():
        agr = by_sys.get("_agreement", {})
        ann = by_sys.get("_annotators", [])
        L += [f"### {g}", "", f"Разметчиков: {len(ann)}. α грамотность: {_fmt(agr.get('grammar'))}, "
              f"α смысл: {_fmt(agr.get('meaning'))}.", "",
              "| Система | Грамотность | Смысл | Оценок |", "|---|---|---|---|"]
        items = [(s, v) for s, v in by_sys.items() if not s.startswith("_")]
        for s, v in sorted(items, key=lambda x: -x[1].get("grammar", {}).get("mean", 0)):
            gr, me = v.get("grammar"), v.get("meaning")
            L.append(f"| {s} | {_m(gr)} | {_m(me)} | {(gr or me or {}).get('n', 0)} |")
        L.append("")
    return L


def _m(x):
    return f"{x['mean']:.2f} [{x['ci_low']:.2f}–{x['ci_high']:.2f}]" if x else "—"


def _fmt(x):
    return "—" if x is None else f"{x:.2f}"

