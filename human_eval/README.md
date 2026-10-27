# Оценка носителями

Всё, что бенчмарк не может посчитать сам: эталонные расшифровки живой речи и оценки грамотности. Разметка идёт в одном HTML-файле `annotate.html`: он работает без интернета и без установки, данные остаются в браузере разметчика, пока тот не сохранит файл.

## Для организатора

1. Соберите пакет.

   Расшифровка эфира (фрагменты без эталона):
   ```bash
   python -m bashbench export-human broadcast-spont-v1 asr --out human_eval/packages/asr-broadcast-r1
   ```
   Оценка перевода на башкирский (100 фраз, ответы всех перечисленных систем):
   ```bash
   python -m bashbench export-human phrasebook-v1 ru-ba nllb-200-distilled-600M yandex-translate google-translate gigachat-2-max \
       --per-system 100 --out human_eval/packages/mt-ru-ba-r1
   ```
   Оценка генерации:
   ```bash
   python -m bashbench export-human gen-v1 gen gigachat-2-max yandexgpt --per-system 56 --out human_eval/packages/gen-r1
   ```
2. Отправьте разметчику `annotate.html` и папку пакета **без `key.json`**: в нём написано, какая система дала какой ответ.
3. Для согласия разметчиков отдайте 20% заданий двум людям: это тот же пакет у второго человека, которого вы попросили сделать первые N заданий.
4. Полученные файлы `<пакет>__<имя>.json` положите рядом с `key.json` и выполните:
   ```bash
   python -m bashbench import-human human_eval/packages/mt-ru-ba-r1/*__*.json
   python -m bashbench score
   ```
   Расшифровки попадут в `testsets/broadcast-spont-v1/data.jsonl` (поле `text`), оценки — в `results/human/`, сводка — в таблицу результатов.

## Для разметчика

1. Откройте `annotate.html` в Chrome, Edge или Firefox.
2. Впишите имя латиницей и выберите папку пакета.
3. Инструкция открывается кнопкой «Инструкция» на первом экране. Её полные версии лежат в [GUIDELINES_ASR.md](GUIDELINES_ASR.md) (расшифровка) и [GUIDELINES_RATING.md](GUIDELINES_RATING.md) (оценка).
4. Работу можно прерывать: всё сохраняется в браузере. Не очищайте историю браузера, пока не сохранили файл.
5. В конце нажмите «Сохранить файл» и отправьте его организатору.

Сколько это занимает: на минуту живой речи уходит 8–12 минут расшифровки; одно задание с четырьмя ответами — 1–2 минуты.
