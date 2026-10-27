import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSETS = ROOT / "testsets"
RESULTS = ROOT / "results"
# Куда скачиваются большие файлы (аудио, исходные корпуса). Не коммитится.
CACHE = Path(os.environ.get("BASHBENCH_CACHE", ROOT / ".cache"))


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def testset_path(name: str) -> Path:
    return TESTSETS / name / "data.jsonl"


def load_testset(name: str) -> list[dict]:
    p = testset_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"Набора {name} нет в {p}. Соберите его: python -m bashbench build {name}"
        )
    return read_jsonl(p)


def safe_name(system: str) -> str:
    """hf:Qwen/Qwen3-1.7B → hf_Qwen_Qwen3-1.7B (имя системы становится именем файла)."""
    return re.sub(r"[^\w.+-]+", "_", system).strip("_")


def run_path(track: str, testset: str, task: str, system: str) -> Path:
    return RESULTS / "runs" / track / testset / task / f"{safe_name(system)}.jsonl"
