"""Тесты векторизованного regex pre-filter."""
import pytest

from app.models.spam.regex_model import SpamRegexModel
from app.models.toxicity.regex_model import RegexModel
from app.utils.regex_batch import batch_regex_classify


def _predict_loop(texts, patterns, empty_fn, hit_fn):
    out = []
    for text in texts:
        if not text or not str(text).strip():
            out.append(empty_fn())
            continue
        matched = {}
        for name, pattern in patterns:
            if pattern.search(str(text)):
                matched[name] = 1.0
        out.append(hit_fn(matched) if matched else empty_fn())
    return out


@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["   "],
        ["нормальный комментарий"],
        ["это блять какой-то текст"],
        ["заработок без вложений от 5000 руб в день"],
        ["нормальный", "токсичный блять", "без вложений заработок"],
    ],
)
def test_batch_regex_classify_matches_loop(texts):
    tox_model = RegexModel()
    patterns = tox_model._labeled_patterns()

    expected = _predict_loop(
        texts,
        patterns,
        empty_fn=RegexModel.empty_result,
        hit_fn=RegexModel._hit_result,
    )
    actual = batch_regex_classify(
        texts,
        patterns,
        empty_result=RegexModel.empty_result,
        hit_result=RegexModel._hit_result,
        min_texts_for_buckets=999_999,  # без бакетов
    )
    assert actual == expected

    actual_bucketed = batch_regex_classify(
        texts,
        patterns,
        empty_result=RegexModel.empty_result,
        hit_result=RegexModel._hit_result,
        min_texts_for_buckets=1,
        length_buckets=4,
    )
    assert actual_bucketed == expected


def test_tox_regex_predict_batch_equivalent_to_predict():
    model = RegexModel()
    model.load()
    texts = [
        "нормальный текст",
        "токсичный блять текст",
        "",
        "идиот",
        "привет мир",
    ]
    batch = model.predict_batch(texts)
    single = [model.predict(t) for t in texts]
    assert batch == single


def test_spam_regex_predict_batch_equivalent_to_predict():
    model = SpamRegexModel()
    model.load()
    texts = [
        "обычный комментарий",
        "заработок без вложений",
        "",
        "перейди по ссылке",
    ]
    batch = model.predict_batch(texts)
    single = [model.predict(t) for t in texts]
    assert batch == single


def test_spam_batch_on_many_texts():
    """Батч и поштучный predict совпадают на случайной выборке длин."""
    model = SpamRegexModel()
    model.load()
    texts = [
        "короткий",
        "x" * 20,
        "заработок без вложений " + "a" * 200,
        "нормальный комментарий " + "b" * 500,
        "t.me/spamchannel",
    ] * 400
    batch = model.predict_batch(texts)
    single = [model.predict(t) for t in texts]
    assert batch == single
