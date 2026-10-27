"""python -m bashbench <команда>

  build <набор>                         собрать тестовый набор
  run <набор> <задача> <система> [-n N] прогнать систему
  score                                 пересчитать метрики и LEADERBOARD.md
  export-human <набор> <задача> ...     выгрузить ответы систем на оценку носителями
"""

import argparse


def main():
    ap = argparse.ArgumentParser(prog="bashbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("testset")

    r = sub.add_parser("run")
    r.add_argument("testset")
    r.add_argument("task")
    r.add_argument("system")
    r.add_argument("-n", "--limit", type=int)

    sub.add_parser("score")

    h = sub.add_parser("export-human")
    h.add_argument("testset")
    h.add_argument("task")
    h.add_argument("systems", nargs="*")
    h.add_argument("--per-system", type=int, default=100)
    h.add_argument("--out", required=True)

    i = sub.add_parser("import-human")
    i.add_argument("files", nargs="+")

    a = ap.parse_args()
    if a.cmd == "build":
        from .data import build

        if a.testset in build.BUILDERS:
            build.BUILDERS[a.testset]()
        else:
            from .data import asr_sets

            asr_sets.BUILDERS[a.testset]()
    elif a.cmd == "run":
        from .run import run

        run(a.testset, a.task, a.system, a.limit)
    elif a.cmd == "score":
        from .score import score_all

        score_all()
    elif a.cmd == "export-human":
        from .human import export

        export(a.testset, a.task, a.systems, a.per_system, a.out)
    elif a.cmd == "import-human":
        from .human import import_files

        import_files(a.files)


if __name__ == "__main__":
    main()
