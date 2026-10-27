# flores200-devtest

FLORES-200 devtest, `bak_Cyrl` ↔ `rus_Cyrl`, 1012 предложений. CC BY-SA 4.0, в репозитории не хранится:

```bash
python -m bashbench build flores200-devtest
```

Ни одно предложение не нашлось дословно в корпусе Айгиза (`contamination.json`). NLLB настраивали на dev-части FLORES, а не на devtest.
