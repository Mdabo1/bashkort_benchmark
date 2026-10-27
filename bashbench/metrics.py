"""Метрики с бутстреп-интервалами.

Все интервалы считаются одинаково: ресэмплинг предложений/записей с возвращением,
корпусная метрика на каждой выборке, 95% перцентильный интервал. Для сравнения
двух систем — парный бутстреп на одних и тех же индексах.
"""

from dataclasses import asdict, dataclass

import jiwer
import numpy as np
import sacrebleu

from .normalize import normalize_for_wer

N_BOOTSTRAP = 1000
SEED = 12345


@dataclass
class Score:
    metric: str
    value: float
    ci_low: float
    ci_high: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _bootstrap(stats: np.ndarray, from_stats, n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(stats)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([from_stats(stats[i].sum(axis=0)) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), idx


# --- перевод -----------------------------------------------------------------

def _sacre(metric_name: str):
    if metric_name == "chrF++":
        return sacrebleu.metrics.CHRF(word_order=2)
    if metric_name == "chrF":
        return sacrebleu.metrics.CHRF()
    if metric_name == "BLEU":
        return sacrebleu.metrics.BLEU()
    raise ValueError(metric_name)


def mt_stats(metric_name: str, hyps: list[str], refs: list[str]) -> tuple[np.ndarray, callable]:
    m = _sacre(metric_name)
    stats = np.array(m._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)
    return stats, lambda s: m._compute_score_from_stats(list(s)).score


def mt_score(metric_name: str, hyps: list[str], refs: list[str]) -> Score:
    stats, f = mt_stats(metric_name, hyps, refs)
    lo, hi, _ = _bootstrap(stats, f)
    return Score(metric_name, f(stats.sum(axis=0)), lo, hi, len(hyps))


# --- распознавание речи ------------------------------------------------------

def asr_stats(hyps: list[str], refs: list[str], unit: str = "word") -> np.ndarray:
    """По строке на запись: [ошибки, длина эталона]."""
    rows = []
    for h, r in zip(hyps, refs):
        h, r = normalize_for_wer(h), normalize_for_wer(r)
        if unit == "char":
            h, r = " ".join(h.replace(" ", "_")), " ".join(r.replace(" ", "_"))
        n_ref = len(r.split())
        if n_ref == 0:
            rows.append([len(h.split()), 0])
            continue
        o = jiwer.process_words(r, h if h else "")
        rows.append([o.substitutions + o.deletions + o.insertions, n_ref])
    return np.array(rows, dtype=np.float64)


def _rate(s: np.ndarray) -> float:
    return 100.0 * s[0] / max(s[1], 1.0)


def asr_score(hyps: list[str], refs: list[str], unit: str = "word") -> Score:
    stats = asr_stats(hyps, refs, unit)
    lo, hi, _ = _bootstrap(stats, _rate)
    return Score("WER" if unit == "word" else "CER", _rate(stats.sum(axis=0)), lo, hi, len(hyps))


# --- парное сравнение --------------------------------------------------------

def paired_bootstrap(stats_a: np.ndarray, stats_b: np.ndarray, from_stats, higher_is_better: bool,
                     n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Доля выборок, где A лучше B, и интервал разницы A − B."""
    rng = np.random.default_rng(seed)
    n = len(stats_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.array([from_stats(stats_a[i].sum(0)) - from_stats(stats_b[i].sum(0)) for i in idx])
    wins = (diffs > 0).mean() if higher_is_better else (diffs < 0).mean()
    return {
        "delta": from_stats(stats_a.sum(0)) - from_stats(stats_b.sum(0)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_a_better": float(wins),
    }


# --- выбор из пары -----------------------------------------------------------

def accuracy(correct: list[bool]) -> Score:
    arr = np.array(correct, dtype=np.float64)[:, None]
    lo, hi, _ = _bootstrap(arr, lambda s: 100.0 * s[0] / len(arr))
    return Score("accuracy", 100.0 * arr.mean(), lo, hi, len(arr))
