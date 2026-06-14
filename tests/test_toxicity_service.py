"""Тесты ToxicityService (prefilter + ML)."""
from app.core.model_manager import ModelManager
from app.services.toxicity.service import ToxicityService


class _RegexStub:
    model_name = "regex"
    is_loaded = True

    def __init__(self) -> None:
        self.predict_batch_calls = 0

    def predict(self, text: str):
        return {"is_toxic": False, "toxicity_score": 0.0, "toxicity_types": {}}

    def predict_batch(self, texts):
        self.predict_batch_calls += 1
        return [self.predict(t) for t in texts]


class _TfidfStub:
    model_name = "tfidf"
    is_loaded = True

    def __init__(self) -> None:
        self.predict_batch_calls = 0

    def predict(self, text: str):
        return {"is_toxic": False, "toxicity_score": 0.1, "toxicity_types": {}}

    def predict_batch(self, texts):
        self.predict_batch_calls += 1
        return [self.predict(t) for t in texts]


def _service_with_models(current: str) -> tuple[ToxicityService, _RegexStub, _TfidfStub]:
    regex = _RegexStub()
    tfidf = _TfidfStub()
    manager = ModelManager(fallback_chain=["tfidf", "regex"])
    manager.register_model("regex", regex)
    manager.register_model("tfidf", tfidf)
    manager.set_current_model(current)
    return ToxicityService(manager), regex, tfidf


def test_classify_batch_skips_second_regex_pass():
    service, regex, tfidf = _service_with_models("regex")
    results = service.classify_batch(["hello", "world"])
    assert regex.predict_batch_calls == 1
    assert tfidf.predict_batch_calls == 0
    assert len(results) == 2
    assert all(r["is_toxic"] is False for r in results)


def test_classify_batch_runs_tfidf_after_regex_prefilter():
    service, regex, tfidf = _service_with_models("tfidf")
    results = service.classify_batch(["hello"])
    assert regex.predict_batch_calls == 1
    assert tfidf.predict_batch_calls == 1
    assert results[0]["tox_model_used"] == "tfidf"
