# cv25-test

Тестовая часть Mozilla Common Voice Scripted Speech 25.0 (ba): 14 572 записи, 19,9 ч чтения по подсказке. CC0.

Common Voice раздаётся через [Mozilla Data Collective](https://datacollective.mozillafoundation.org/) после регистрации, поэтому набор собирается локально:

```bash
export CV25_BA_DIR=/путь/к/cv-corpus-25.0/ba   # там test.tsv и clips/
python -m bashbench build cv25-test
```

Модели, обученные на train-части CV (GigaAM-Bashkir, MMS, Omnilingual и др.), видели тех же дикторов в других записях. Поэтому этот набор показывает качество на чтении в знакомых условиях, а не на живой речи.
