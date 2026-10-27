#!/usr/bin/env bash
# Все бейзлайны с открытыми весами. На CPU (4 ядра) — несколько часов, на GPU — минуты.
set -euo pipefail
cd "$(dirname "$0")/.."
B="python -m bashbench"

for ts in phrasebook-v1 flores200-devtest; do
  for d in ba-ru ru-ba; do
    $B run $ts $d nllb-200-distilled-600M
  done
done

$B run orth-pairs-v1 choice goldfish-bak
$B run orth-pairs-v1 choice hf:Qwen/Qwen3-1.7B   # на CPU около 2 ч, на GPU — минуты

for s in gigaam-bashkir-cv25-int8 w2v-bert-2.0-bashkort-russian-omnivoice mhubert-147-bashkort mms-1b-all; do
  $B run broadcast-spont-v1 asr $s
done

$B score
