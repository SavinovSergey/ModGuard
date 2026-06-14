"""Режимы moderation_pipeline в ClassificationService."""
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services.classification import ClassificationService, _stub_spam_batch, _stub_tox_batch


@pytest.fixture
def classification_service():
    service = ClassificationService(model_manager=MagicMock(), spam_model=None, spam_regex_model=None)
    service._timed_toxicity_batch = MagicMock(
        return_value=(
            [
                {
                    "is_toxic": True,
                    "toxicity_score": 0.9,
                    "toxicity_types": {"хуй": 1.0},
                    "tox_model_used": "tfidf",
                }
            ],
            10.0,
        )
    )
    service._timed_spam_batch = MagicMock(
        return_value=(
            [{"is_spam": True, "spam_score": 0.8, "spam_model_used": "tfidf"}],
            20.0,
        )
    )
    return service


def test_tox_only_runs_toxicity_and_spam_stub(classification_service, monkeypatch):
    monkeypatch.setattr(settings, "moderation_pipeline", "tox_only")

    results = classification_service.classify_batch(["текст"])

    classification_service._timed_toxicity_batch.assert_called_once()
    classification_service._timed_spam_batch.assert_not_called()
    assert results[0]["is_toxic"] is True
    assert results[0]["tox_model_used"] == "tfidf"
    assert results[0]["is_spam"] is False
    assert results[0]["spam_score"] == 0.0
    assert results[0]["spam_model_used"] is None


def test_spam_only_runs_spam_and_tox_stub(classification_service, monkeypatch):
    monkeypatch.setattr(settings, "moderation_pipeline", "spam_only")

    results = classification_service.classify_batch(["текст"])

    classification_service._timed_spam_batch.assert_called_once()
    classification_service._timed_toxicity_batch.assert_not_called()
    assert results[0]["is_spam"] is True
    assert results[0]["spam_model_used"] == "tfidf"
    assert results[0]["is_toxic"] is False
    assert results[0]["toxicity_score"] == 0.0
    assert results[0]["tox_model_used"] is None


def test_both_runs_parallel(classification_service, monkeypatch):
    monkeypatch.setattr(settings, "moderation_pipeline", "both")
    classification_service._use_process_pool = False

    results = classification_service.classify_batch(["текст"])

    classification_service._timed_toxicity_batch.assert_called_once()
    classification_service._timed_spam_batch.assert_called_once()
    assert results[0]["is_toxic"] is True
    assert results[0]["is_spam"] is True


def test_stub_batches_have_expected_shape():
    tox = _stub_tox_batch(2)
    spam = _stub_spam_batch(2)
    assert len(tox) == 2
    assert len(spam) == 2
    assert tox[0]["toxicity_types"] == {}
    assert spam[1]["spam_model_used"] is None
