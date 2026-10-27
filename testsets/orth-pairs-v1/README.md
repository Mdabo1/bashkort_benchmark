# orth-pairs-v1

1000 минимальных пар «правильное предложение / то же с одной ошибкой»: 500 `letters` (башкирские буквы заменены русскими) и 500 `harmony` (нарушен сингармонизм в окончании).

- Предложения: [bashkorttele/periodicals-izddom](https://huggingface.co/datasets/bashkorttele/periodicals-izddom) @ `c18e45e9`, CC BY 4.0, только записи с `text_source == "native"`.
- Построение и фильтры: `bashbench/data/build.py: build_orth_pairs`, описание — `docs/methodology.md`.
- Поля: `good`, `bad`, `kind`, `word_good`, `word_bad`, `freq_good` и `freq_bad` (частоты слов в периодике), `source_id`, `verified`.
- `verified: false`: носители пары пока не проверяли. Поле `bad` содержит ошибку **намеренно**.

Атрибуция обязательна, см. `NOTICE` в корне репозитория.
