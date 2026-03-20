from unittest.mock import MagicMock, patch


def test_validate_toxicity_rnn_accepts_hf_model_id():
    # Import внутри теста, чтобы убедиться что файл парсится без запуска сети
    from scripts.toxicity.validate_toxicity import _load_model

    hf_id = "SergeySavinov/rurnn-toxicity"

    mock_model = MagicMock()
    mock_model.load = MagicMock()

    with patch("app.models.toxicity.rnn_model.RNNModel", return_value=mock_model) as mock_cls:
        model = _load_model("rnn", hf_id)

        assert model is mock_model
        # В HF ветке должны передать hf_model_name
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("hf_model_name") == hf_id
        mock_model.load.assert_called_once_with()

