# phrasebook-v1

1244 разговорные фразы ba↔ru из пяти разговорников БГПУ им. М. Акмуллы.

- Источник: [bashkorttele/trilingual-parallel-phrasebooks-bgpu](https://huggingface.co/datasets/bashkorttele/trilingual-parallel-phrasebooks-bgpu), CC BY 4.0, через исправленную редакцию [failed09/bashkir-multilingual-phrasebooks](https://huggingface.co/datasets/failed09/bashkir-multilingual-phrasebooks) @ `eae6488f`.
- Отбор: `entry_type == "phrase"`, без скобок, косых черт и многоточий, не короче двух слов, без повторов после нормализации (`bashbench/data/build.py: build_phrasebook`).
- Поля: `id`, `ba`, `ru`, `source_book`, `in_aigiz_corpus_ba`, `in_aigiz_corpus_ru`. Последние два поля показывают, есть ли сторона дословно в [AigizK/bashkir-russian-parallel-corpora](https://huggingface.co/datasets/AigizK/bashkir-russian-parallel-corpora) (см. `contamination.json`). Такие пары есть у 122 из 1244 башкирских сторон.
- Эталон один, носители его не проверяли. У фразы бывает несколько правильных переводов, поэтому автоматические метрики на коротких фразах шумные: смотрите на интервалы.

Атрибуция обязательна, см. `NOTICE` в корне репозитория.
