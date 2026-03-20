import pytest
from unittest.mock import MagicMock, patch


def test_quantize_bert_to_onnx_cpu_accepts_hf_model_id(tmp_path):
    """
    Проверяет, что quantize_bert_onnx.py умеет принимать HF model-id (вместо локальной папки),
    как минимум в режиме export_only=True (без тяжелой квантизации).
    """

    from scripts.toxicity import quantize_bert_onnx as q

    mock_ort_model = MagicMock()
    mock_ort_model.save_pretrained = MagicMock()

    class _MockORTModel:
        @staticmethod
        def from_pretrained(model_id, export=True):
            assert export is True
            assert model_id == "SergeySavinov/rubert-tiny-toxicity"
            return mock_ort_model

    mock_tokenizer = MagicMock()
    mock_tokenizer.save_pretrained = MagicMock()

    class _MockORTQuantizer:
        @staticmethod
        def from_pretrained(_):
            return MagicMock()

    mock_auto_qconfig = MagicMock()

    with patch.object(q, "_ensure_optimum_onnx", return_value=(_MockORTModel, _MockORTQuantizer, mock_auto_qconfig)):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as tok_call:
            out_dir = tmp_path / "onnx_out"
            res = q.quantize_bert_to_onnx_cpu(
                "SergeySavinov/rubert-tiny-toxicity",
                out_dir,
                export_only=True,
            )

            assert res == out_dir
            tok_call.assert_called_once_with("SergeySavinov/rubert-tiny-toxicity")
            mock_ort_model.save_pretrained.assert_called_once_with(str(out_dir))
            mock_tokenizer.save_pretrained.assert_called_once_with(str(out_dir))

