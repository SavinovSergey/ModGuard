import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("pandas", reason="pandas не установлен")
pytest.importorskip("matplotlib", reason="matplotlib не установлен")


def test_validate_toxicity_bert_accepts_hf_model_id():
    from scripts.toxicity.validate_toxicity import _load_model

    hf_id = "SergeySavinov/rubert-tiny-toxicity"

    mock_model = MagicMock()
    mock_model.load = MagicMock()

    with patch("app.models.toxicity.bert_model.BERTModel", return_value=mock_model) as mock_bert_cls:
        model = _load_model("bert", hf_id)

        assert model is mock_model
        # В HF-ветке конструктор должен получить model_name
        _, kwargs = mock_bert_cls.call_args
        assert kwargs.get("model_name") == hf_id
        mock_model.load.assert_called_once_with()

