"""Подсчёт метрик по всем прогонам и сборка results/LEADERBOARD.md."""

import statistics
from collections import defaultdict
from pathlib import Path

from . import metrics
from .io import RESULTS, load_testset, read_jsonl, write_json


def _runs():
    for p in sorted((RESULTS / "runs").glob("*/*/*/*.jsonl")):
        track, testset, task = p.parts[-4:-1]
        yield track, testset, task, p.stem, p


def score_mt(testset, task, rows):
    src, tgt = task.split("-")
    gold = {it["id"]: it for it in load_testset(testset)}
    rows = [r for r in rows if r["id"] in gold]
    hyps = [r["hyp"] for r in rows]
    refs = [gold[r["id"]][tgt] for r in rows]
    out = {m: metrics.mt_score(m, hyps, refs).as_dict() for m in ("chrF++", "BLEU")}
    # без пар, которые дословно есть в корпусе Айгиза (основной открытый корпус для обучения)
    flag = f"in_aigiz_corpus_{src}"
    clean = [i for i, r in enumerate(rows) if not gold[r["id"]].get(flag, False)]
    if len(clean) < len(rows):
        out["chrF++_clean"] = metrics.mt_score("chrF++", [hyps[i] for i in clean], [refs[i] for i in clean]).as_dict()
    return out


def score_asr(testset, rows):
    gold = {it["id"]: it for it in load_testset(testset)}
    rows = [r for r in rows if r["id"] in gold and gold[r["id"]].get("text")]
    if not rows:
        return {}
    hyps, refs = [r["hyp"] for r in rows], [gold[r["id"]]["text"] for r in rows]
    return {"WER": metrics.asr_score(hyps, refs).as_dict(), "CER": metrics.asr_score(hyps, refs, "char").as_dict()}


def score_choice(testset, rows):
    gold = {it["id"]: it for it in load_testset(testset)}
    out = {"all": metrics.accuracy([r["correct"] for r in rows]).as_dict()}
    by = defaultdict(list)
    for r in rows:
        by[gold[r["id"]]["kind"]].append(r["correct"])
    for k, v in by.items():
        out[k] = metrics.accuracy(v).as_dict()
    if rows and rows[0].get("method") == "prompt":
        out["unparsed"] = sum(r["pred"] is None for r in rows)
    out["method"] = rows[0].get("method") if rows else None
    return out


def score_gen(rows):
    from .orthography import analyze

    an = [analyze(r["hyp"]) for r in rows]
    keys = ["bashkir_letter_words", "foreign_letter_words", "non_alphabet_words", "oov_words", "latin_words"]
    out = {k: 100 * statistics.mean(a[k] for a in an) for k in keys}
    out["lang_ba"] = 100 * statistics.mean(a["lang"] == "ba" for a in an)
    out["lang_other"] = dict(sorted(_count(a["lang"] for a in an if a["lang"] != "ba").items(), key=lambda x: -x[1]))
    out["median_words"] = statistics.median(a["n_words"] for a in an)
    out["n"] = len(an)
    return out


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return dict(d)


def score_all():
    scores = defaultdict(dict)
    for track, testset, task, system, path in _runs():
        rows = read_jsonl(path)
        if track == "mt":
            s = score_mt(testset, task, rows)
        elif track == "asr":
            s = score_asr(testset, rows)
        elif task == "choice":
            s = score_choice(testset, rows)
        else:
            s = score_gen(rows)
        if s:
            scores[f"{track}/{testset}/{task}"][system] = s
    human = RESULTS / "human" / "summary.json"
    write_json(RESULTS / "scores.json", scores)
    leaderboard(scores, human)
    print(f"→ {RESULTS / 'LEADERBOARD.md'}")


def _ci(s, fmt="{:.1f}"):
    return f"{fmt.format(s['value'])} [{fmt.format(s['ci_low'])}–{fmt.format(s['ci_high'])}]"


def leaderboard(scores, human_path: Path):
    L = ["# Таблица результатов", "",
         "Сгенерировано `python -m bashbench score`. В квадратных скобках — 95% бутстреп-интервал. ",
         "Если интервалы двух систем перекрываются, разница может быть случайной.", ""]

    mt_sets = sorted({k.split("/")[1] for k in scores if k.startswith("mt/")})
    if mt_sets:
        L += ["## Перевод", "", "Главная метрика — chrF++ (выше — лучше). BLEU для сравнения со статьями.", ""]
    for ts in mt_sets:
        systems = sorted({s for k, v in scores.items() if k.startswith(f"mt/{ts}/") for s in v})
        has_clean = any("chrF++_clean" in v.get(s, {}) for k, v in scores.items() if k.startswith(f"mt/{ts}/") for s in v)
        head = "| Система | ba→ru chrF++ | ru→ba chrF++ | ba→ru BLEU | ru→ba BLEU |" + (" ba→ru chrF++ без пересечений | ru→ba chrF++ без пересечений |" if has_clean else "")
        L += [f"### {ts}", "", head, "|" + "---|" * (head.count("|") - 1)]

        def key(s):
            v = scores.get(f"mt/{ts}/ru-ba", {}).get(s) or scores.get(f"mt/{ts}/ba-ru", {}).get(s)
            return -v["chrF++"]["value"]

        for s in sorted(systems, key=key):
            a, b = scores.get(f"mt/{ts}/ba-ru", {}).get(s), scores.get(f"mt/{ts}/ru-ba", {}).get(s)
            row = [s, _ci(a["chrF++"]) if a else "—", _ci(b["chrF++"]) if b else "—",
                   f"{a['BLEU']['value']:.1f}" if a else "—", f"{b['BLEU']['value']:.1f}" if b else "—"]
            if has_clean:
                row += [_ci(a["chrF++_clean"]) if a and "chrF++_clean" in a else "—",
                        _ci(b["chrF++_clean"]) if b and "chrF++_clean" in b else "—"]
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    asr = {k: v for k, v in scores.items() if k.startswith("asr/")}
    if asr:
        L += ["## Распознавание речи", "", "WER и CER в процентах (ниже — лучше), после нормализации из `bashbench/normalize.py`.", ""]
        for k, v in sorted(asr.items()):
            L += [f"### {k.split('/')[1]}", "", "| Система | WER | CER | Записей |", "|---|---|---|---|"]
            for s, m in sorted(v.items(), key=lambda x: x[1]["WER"]["value"]):
                L.append(f"| {s} | {_ci(m['WER'])} | {_ci(m['CER'])} | {m['WER']['n']} |")
            L.append("")

    ch = scores.get("llm/orth-pairs-v1/choice")
    if ch:
        L += ["## Грамотность: минимальные пары", "",
              "Доля пар, где система выбрала правильный вариант (случайное угадывание — 50%). "
              "`letters`: башкирские буквы заменены русскими (ҡ→к, ҙ→з…); `harmony`: нарушен сингармонизм в окончании. "
              "`logprob` — сравнение вероятностей (открытые веса), `prompt` — модель спрашивают, какой вариант верный.", "",
              "| Система | Метод | Все | letters | harmony |", "|---|---|---|---|---|"]
        for s, m in sorted(ch.items(), key=lambda x: -x[1]["all"]["value"]):
            L.append(f"| {s} | {m['method']} | {_ci(m['all'])} | {_ci(m['letters']) if 'letters' in m else '—'} | "
                     f"{_ci(m['harmony']) if 'harmony' in m else '—'} |")
        L.append("")

    gen = scores.get("llm/gen-v1/gen")
    if gen:
        L += ["## Генерация на башкирском: автоматические сигналы", "",
              "Это фильтр грубых провалов, а не оценка качества: качество оценивают носители (раздел ниже). "
              "«По-башкирски» — доля ответов, которые fastText lid.176 определил как башкирский. "
              "«Слова с башкирскими буквами»: в периодике bashkorttele их 66%. «Чужие буквы» — доля слов с қ, ұ, і, җ и т. п. «Нет в корпусе» — доля слов, которых нет в частотном "
              "словаре периодики bashkorttele.", "",
              "| Система | По-башкирски, % | Слова с башкирскими буквами, % | Чужие буквы, % | Нет в корпусе, % | Медиана слов |",
              "|---|---|---|---|---|---|"]
        for s, m in sorted(gen.items(), key=lambda x: -x[1]["lang_ba"]):
            L.append(f"| {s} | {m['lang_ba']:.0f} | {m['bashkir_letter_words']:.0f} | {m['foreign_letter_words']:.1f} | "
                     f"{m['oov_words']:.0f} | {m['median_words']:.0f} |")
        L.append("")

    if human_path.exists():
        from .human import leaderboard_section

        L += leaderboard_section(human_path)

    (RESULTS / "LEADERBOARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
