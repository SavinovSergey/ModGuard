"""
Скрипт динамической квантизации RNN модели для классификации токсичности.

Использует torch.quantization.quantize_dynamic (или torch.ao.quantization.quantize_dynamic)
для квантизации весов в int8. Автоматически выбирает правильный импорт в зависимости от версии PyTorch.
Подходит для RNN моделей (RNN, GRU, LSTM).

Использование:
  1. Отдельный скрипт:
     python scripts/toxicity/quantize_rnn.py models/toxicity/rnn/model.pt models/toxicity/rnn/tokenizer.json -o models/toxicity/rnn

  2. Импорт после обучения:
     from scripts.toxicity.quantize_rnn import quantize_rnn_model
     quantize_rnn_model("models/toxicity/rnn/model.pt", "models/toxicity/rnn/tokenizer.json", "models/toxicity/rnn")
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Union

import torch

# Импорт функции квантизации с поддержкой разных версий PyTorch
try:
    # PyTorch 1.13+ (новый путь)
    from torch.ao.quantization import quantize_dynamic
    _QUANTIZATION_METHOD = "torch.ao.quantization.quantize_dynamic"
except ImportError:
    # PyTorch < 1.13 (старый путь, fallback)
    _QUANTIZATION_METHOD = "torch.quantization.quantize_dynamic"
    from torch.quantization import quantize_dynamic

import torch.nn as nn

_DEFAULT_QUANTIZE_DTYPE = torch.qint8

import sys
# Корень проекта (скрипт в scripts/toxicity/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.toxicity.rnn_network import RNNClassifier
from app.models.toxicity.rnn_tokenizers import create_tokenizer


def quantize_rnn_model(
    model_path: Union[str, Path],
    tokenizer_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    device: str = "cpu",
    dtype: "torch.dtype" = None,
) -> Path:
    """
    Динамическая квантизация RNN модели.

    Args:
        model_path: Путь к файлу модели (.pt)
        tokenizer_path: Путь к файлу токенизатора (.json)
        output_dir: Директория для сохранения квантизированной модели
        device: Устройство для загрузки модели ('cpu' или 'cuda')
        dtype: Тип квантизации (torch.qint8 или torch.float16)

    Returns:
        Путь к сохранённой квантизированной модели
    """
    # Устанавливаем dtype по умолчанию если не указан (используем константу из модуля)
    if dtype is None:
        dtype = _DEFAULT_QUANTIZE_DTYPE
    
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Токенизатор не найден: {tokenizer_path}")

    # Загружаем параметры модели
    params_path = model_path.parent / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Файл параметров не найден: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        model_params = json.load(f)

    print(f"Загрузка модели из {model_path}...")
    print(f"Параметры модели: {model_params.get('rnn_type', 'gru')}, "
          f"hidden_size={model_params.get('hidden_size', 128)}, "
          f"embedding_dim={model_params.get('embedding_dim', 100)}")

    # Загружаем токенизатор
    tokenizer_type = model_params.get("tokenizer_type", "bpe")
    tokenizer = create_tokenizer(tokenizer_type)
    tokenizer.load(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    # Создаём модель с правильными параметрами
    model = RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=model_params.get("embedding_dim", 100),
        hidden_size=model_params.get("hidden_size", 128),
        num_layers=model_params.get("num_layers", 1),
        rnn_type=model_params.get("rnn_type", "gru"),
        dropout=model_params.get("dropout", 0.2),
        bidirectional=model_params.get("bidirectional", False),
        padding_idx=tokenizer.get_pad_token_id(),
        embedding_dropout=model_params.get("embedding_dropout", 0.0),
        use_layer_norm=model_params.get("use_layer_norm", False),
    )

    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Применение динамической квантизации (dtype={dtype})...")

    # Динамическая квантизация
    # quantize_dynamic квантизирует только Linear и LSTM/GRU слои
    # Embedding слои остаются в float32
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.RNN},
        dtype=dtype,
    )

    # Квантизация эмбеддингов в FP16
    embeddings_quantized = False
    print("Применение квантизации эмбеддингов в FP16...")
    # Преобразуем веса эмбеддингов в float16
    with torch.no_grad():
        quantized_model.embedding.weight.data = quantized_model.embedding.weight.data.half()
    embeddings_quantized = True
    print("Эмбеддинги квантизированы в FP16")

    # Сохраняем квантизированную модель
    quantized_model_path = output_dir / "model_quantized.pt"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"Квантизированная модель сохранена: {quantized_model_path}")

    # Копируем токенизатор
    tokenizer_output_path = output_dir / tokenizer_path.name
    shutil.copy(tokenizer_path, tokenizer_output_path)
    print(f"Токенизатор скопирован: {tokenizer_output_path}")

    # Копируем params.json
    params_output_path = output_dir / "params.json"
    shutil.copy(params_path, params_output_path)
    print(f"Параметры скопированы: {params_output_path}")

    # Добавляем информацию о квантизации в params.json
    with open(params_output_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    params["quantized"] = True
    params["quantization_dtype"] = str(dtype)
    params["quantization_method"] = _QUANTIZATION_METHOD
    params["embeddings_quantized_fp16"] = embeddings_quantized

    with open(params_output_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\nКвантизация завершена. Модель сохранена в {output_dir}")
    print(f"  Размер оригинальной модели: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Размер квантизированной модели: {quantized_model_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Динамическая квантизация RNN модели для классификации токсичности"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Путь к файлу модели (.pt)",
    )
    parser.add_argument(
        "tokenizer_path",
        type=str,
        help="Путь к файлу токенизатора (.json)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Директория для сохранения квантизированной модели (по умолчанию: <model_dir>_quantized)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Устройство для загрузки модели",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["qint8", "float16"],
        default="qint8",
        help="Тип квантизации: qint8 (int8) или float16 (FP16)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)

    if not model_path.exists():
        parser.error(f"Модель не найдена: {model_path}")
    if not tokenizer_path.exists():
        parser.error(f"Токенизатор не найден: {tokenizer_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"{model_path.parent}_quantized"
    output_dir = Path(output_dir)

    dtype_map = {
        "qint8": torch.qint8,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    quantize_rnn_model(
        model_path,
        tokenizer_path,
        output_dir,
        device=args.device,
        dtype=dtype,
    )

    print("Готово.")


if __name__ == "__main__":
    main()
