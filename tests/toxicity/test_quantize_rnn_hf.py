import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="PyTorch не установлен — тест квантизации RNN пропущен")


def test_quantize_rnn_from_hf_downloads_and_calls_quantize(tmp_path):
    """
    Проверяет HF-path для quantize_rnn_from_hf:
    - hf_hub_download возвращает существующие файлы
    - quantize_rnn_model вызывается с путями на локальные файлы в tmpdir
    """
    from scripts.toxicity import quantize_rnn as q

    hf_id = "SergeySavinov/rurnn-toxicity"
    out_dir = tmp_path / "out"

    hf_cache = tmp_path / "hf_cache"
    hf_cache.mkdir()

    def _fake_hf_hub_download(*, repo_id: str, filename: str, **kwargs):
        assert repo_id == hf_id
        target = hf_cache / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename == "params.json":
            target.write_text(
                json.dumps(
                    {
                        "tokenizer_type": "bpe",
                        "rnn_type": "gru",
                        "embedding_dim": 4,
                        "hidden_size": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "bidirectional": False,
                        "max_length": 32,
                        "vocab_size": 10,
                        "embedding_dropout": 0.0,
                        "use_layer_norm": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        else:
            target.write_bytes(b"dummy")
        return str(target)

    quant_stub = MagicMock(return_value=out_dir)

    with patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf_hub_download):
        with patch.object(q, "quantize_rnn_model", quant_stub):
            res = q.quantize_rnn_from_hf(
                hf_id,
                out_dir,
                device="cpu",
            )

            assert res == out_dir
            assert quant_stub.call_count == 1

