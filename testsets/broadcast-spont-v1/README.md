# broadcast-spont-v1

Живая башкирская речь из 22 теле- и радиопрограмм: интервью, новости, ток-шоу, детские передачи. Это основной ASR-набор бенчмарка.

- Аудио: [bashkorttele/broadcast-speech](https://huggingface.co/datasets/bashkorttele/broadcast-speech) @ `5d9dc895`, CDLA-Permissive-2.0. В репозитории хранятся только границы фрагментов (`audio`, `start`, `end`), аудио скачивается при сборке.
- Отбор: Silero VAD, фрагменты 4–20 с, до 2 записей на программу и до 7 фрагментов на запись. ASR-системы в отборе не участвуют. Подробности — `docs/methodology.md`, итог — `stats.json`.
- Эталоны (`text`) пишут носители с нуля по `human_eval/GUIDELINES_ASR.md`. Пока поле пустое, фрагмент в WER не считается.
- `annotations` — все расшифровки с именами разметчиков и флажками; `agreement.json` — расхождение между разметчиками.

```bash
python -m bashbench build broadcast-spont-v1    # скачает ~1 ГБ аудио в .cache/
```
