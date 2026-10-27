# Башкирский бенчмарк

Открытое сравнение ИИ-систем для башкирского языка: перевода, распознавания речи и языковых моделей. Яндекс, Google, ГигаЧат, Omnilingual, NLLB и модели сообщества прогоняются на одних и тех же данных и считаются одинаково. Там, где автоматические метрики врут, качество оценивают носители языка.

Пока «башкирский поддерживается» есть почти у всех, но никто не публикует, **насколько хорошо**. Этот репозиторий закрывает этот пробел.

**Результаты:** [results/LEADERBOARD.md](results/LEADERBOARD.md) · **Методика:** [docs/methodology.md](docs/methodology.md) · **Разметка носителями:** [human_eval/](human_eval/)

## Что измеряется

| Дорожка | Набор | Размер | Что проверяет | Метрика | Статус |
|---|---|---|---|---|---|
| Перевод | `flores200-devtest` | 1012 предл. | книжный текст, ba↔ru | chrF++, BLEU | ✅ готов |
| Перевод | `phrasebook-v1` | 1244 фразы | разговорные фразы, ba↔ru | chrF++, BLEU + носители для ru→ba | ✅ готов |
| Речь | `cv25-test` | 14 572 записи | чтение по подсказке (Common Voice 25) | WER, CER | ✅ сборщик, данные качаются вручную |
| Речь | `broadcast-spont-v1` | 254 фрагмента, 57,5 мин | **живая речь**: интервью, эфир, 21 программа | WER, CER | ⏳ фрагменты выбраны, нужны расшифровки носителей |
| LLM | `orth-pairs-v1` | 1000 пар | узнаёт ли модель правильное написание (ҡ/к, сингармонизм) | точность | ✅ готов, ⏳ выборочная проверка носителями |
| LLM | `gen-v1` | 56 заданий | умеет ли модель писать по-башкирски | авто-сигналы + носители | ✅ задания готовы |

## Системы

| Тип | Системы | Что нужно |
|---|---|---|
| Перевод | `nllb-200-distilled-600M`, `nllb-200-distilled-1.3B`, `nllb-200-3.3B` | ничего, скачивается с HF |
| Перевод | `nllb-200-distilled-600M-ct2int8`, `nllb-200-distilled-1.3B-ct2int8` | то же через CTranslate2 int8: на CPU в разы быстрее |
| Перевод | `yandex-translate` | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` |
| Перевод | `google-translate` | `GOOGLE_API_KEY` |
| Перевод, LLM | `gigachat-2-max`, `gigachat-2-pro` | `GIGACHAT_AUTH_KEY` (+ `GIGACHAT_CA_BUNDLE` при ошибке сертификата) |
| Перевод, LLM | `yandexgpt`, `yandexgpt-lite` | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` |
| Перевод, LLM | `openai:<base_url>\|<model>\|<KEY_ENV>` | любой OpenAI-совместимый сервер: OpenRouter, vLLM и т. п. |
| LLM | `hf:<model_id>`, `goldfish-bak` | открытые веса, считаются log-вероятности |
| Речь | `gigaam-bashkir-cv25-int8`, `gigaam-bashkir-cv25` | ничего (модель Айгиза, ONNX) |
| Речь | `embeddinggemma2-bashkir-cv25`, `w2v-bert-2.0-bashkort-russian-omnivoice`, `mhubert-147-bashkort`, `bashkir-whisper-small`, `whisper-medium-ba` | ничего (модели Айгиза с HF) |
| Речь | `mms-1b-all` | ничего (Meta, CC BY-NC) |
| Речь | `omniasr-llm-7b`, `omniasr-ctc-1b` | `pip install omnilingual-asr`, лучше с GPU |
| Речь | `yandex-speechkit` | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, при необходимости `YANDEX_STT_LANG` |
| Речь | `speechmatics` | `SPEECHMATICS_API_KEY` |

Адаптеры закрытых API написаны по документации, но **ещё не запускались**: ключей у автора первой версии не было. Если адаптер падает, напишите в issues или пришлите исправление.

## Быстрый старт

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[models,lid,dev]"          # torch лучше ставить заранее под вашу платформу

# наборы (то, что лежит в репозитории, пересобирать не нужно)
python -m bashbench build flores200-devtest
python -m bashbench build broadcast-spont-v1  # ~1 ГБ аудио в .cache/

# прогоны: <набор> <задача> <система>
python -m bashbench run phrasebook-v1 ru-ba nllb-200-distilled-600M
python -m bashbench run phrasebook-v1 ru-ba yandex-translate
python -m bashbench run orth-pairs-v1 choice gigachat-2-max
python -m bashbench run gen-v1 gen gigachat-2-max
python -m bashbench run broadcast-spont-v1 asr gigaam-bashkir-cv25-int8

python -m bashbench score                     # → results/LEADERBOARD.md
```

Прогон дописывается порциями: если API упал на середине, повторный запуск продолжит с места остановки. Все открытые бейзлайны: `scripts/run_open_baselines.sh`.

## Принципы

- **Воспроизводимость.** Источники закреплены по коммиту на Hugging Face, у каждого прогона есть `.meta.json` с хешем набора и версией кода.
- **Интервалы, а не только числа.** У каждой метрики 95% бутстреп-интервал.
- **Проверка утечек.** Пересечения с корпусом Айгиза (1,2 млн пар ba–ru) помечены, метрика на «чистом» подмножестве публикуется отдельно.
- **Вслепую.** Носители не знают, какая система дала ответ. Эталоны речи пишутся с нуля, без подсказок от ASR.
- **Только открытые данные.** Тексты и аудио из серии [Bashkorttele](https://huggingface.co/bashkorttele), FLORES-200 и Common Voice.

## Как помочь

1. **Носителям языка:** расшифровать 15–30 минут эфира или оценить 100 переводов. Как это устроено — в [human_eval/README.md](human_eval/README.md), на всё нужен только браузер.
2. **С ключами API:** прогнать Яндекс, Google, ГигаЧат, YandexGPT и прислать `results/runs/...` пулл-реквестом.
3. **С GPU:** Omnilingual ASR 7B, NLLB 3.3B, открытые LLM на 7–70 млрд параметров.

## Лицензии

Код — MIT. Наборы — на условиях источников (CC BY 4.0, CDLA-Permissive-2.0, CC BY-SA 4.0), атрибуция в [NOTICE](NOTICE) и `testsets/*/README.md`.
