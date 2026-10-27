from bashbench import metrics
from bashbench.human import krippendorff_alpha_interval
from bashbench.normalize import fix_homoglyphs, normalize_for_wer
from bashbench.run import _parse_choice


def test_normalize_punct_case_dashes():
    assert normalize_for_wer("Һаумыһығыҙ, дуҫтар!  Ҡайһы-бер — кеше.") == "һаумыһығыҙ дуҫтар ҡайһы бер кеше"


def test_homoglyphs_only_in_cyrillic_words():
    # латинская ə и c в кириллическом слове → кириллица; чисто латинское слово не трогаем
    assert fix_homoglyphs("бəлəкəй cәләм hello") == "бәләкәй сәләм hello"


def test_wrong_letters_are_errors_not_normalized():
    # к вместо ҡ — настоящая ошибка, нормализация её не прощает
    assert normalize_for_wer("кыҙ") != normalize_for_wer("ҡыҙ")
    assert normalize_for_wer("қыҙ") != normalize_for_wer("ҡыҙ")


def test_wer_basic():
    s = metrics.asr_score(["мин мәктәпкә барам"], ["Мин мәктәпкә барам."])
    assert s.value == 0.0
    s = metrics.asr_score(["мин мәктәпкә"], ["мин мәктәпкә барам"])
    assert abs(s.value - 100 / 3) < 1e-6


def test_wer_is_corpus_level():
    # корпусный WER = сумма ошибок / сумма слов, а не среднее по записям
    s = metrics.asr_score(["а", "б в г д"], ["х", "б в г д"])
    assert abs(s.value - 20.0) < 1e-6


def test_chrf_matches_sacrebleu():
    import sacrebleu

    hyps = ["Беҙ бесән сабабыҙ", "Мин асҡысты юғалттым."]
    refs = ["Беҙ бесән сабабыҙ.", "Мин асҡысымды юғалттым."]
    ours = metrics.mt_score("chrF++", hyps, refs)
    ref = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    assert abs(ours.value - ref) < 1e-9
    assert ours.ci_low <= ours.value <= ours.ci_high


def test_parse_choice():
    assert _parse_choice("A") == "A"
    assert _parse_choice(" **B**.") == "B"
    assert _parse_choice("А") == "A"  # кириллическая А
    assert _parse_choice("Вариант А") == "A"
    assert _parse_choice("не знаю") is None


def test_alpha():
    assert krippendorff_alpha_interval({"a": [5, 5], "b": [1, 1], "c": [3, 3]}) == 1.0
    assert krippendorff_alpha_interval({"a": [5, 1], "b": [1, 5]}) < 0
