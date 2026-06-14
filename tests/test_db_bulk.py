"""Unit-тесты bulk-хелперов Postgres (без живой БД)."""
from app.core.db import _result_to_row, _results_to_update_arrays


def test_results_to_update_arrays_shape():
    results = [
        {
            "is_toxic": True,
            "toxicity_score": 0.9,
            "toxicity_types": {"хуй": 1.0},
            "tox_model_used": "regex",
            "spam_model_used": "tfidf",
            "is_spam": False,
            "spam_score": 0.1,
        },
        {
            "is_toxic": False,
            "toxicity_score": 0.0,
            "toxicity_types": {},
            "tox_model_used": "tfidf",
            "spam_model_used": None,
            "is_spam": True,
            "spam_score": 0.95,
            "error": None,
        },
    ]
    arrays = _results_to_update_arrays(results, from_cache=True)
    assert len(arrays) == 10
    from_cache_a, tox_model_a, spam_model_a, is_toxic_a, tox_score_a, tox_types_a, is_spam_a, spam_score_a, error_a, item_index_a = arrays
    assert from_cache_a == [True, True]
    assert item_index_a == [0, 1]
    assert tox_model_a == ["regex", "tfidf"]
    assert is_toxic_a == [True, False]
    assert is_spam_a == [False, True]
    assert tox_types_a[0] == '{"хуй": 1.0}'
    assert tox_types_a[1] == "{}"


def test_result_to_row_matches_legacy_tuple():
    result = {"is_toxic": False, "toxicity_score": 0.2, "toxicity_types": {"a": 1}}
    assert _result_to_row(result, from_cache=False) == (
        False,
        None,
        None,
        False,
        0.2,
        '{"a": 1}',
        False,
        0.0,
        None,
    )
