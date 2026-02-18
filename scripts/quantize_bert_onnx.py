"""
Скрипт ONNX квантизации BERT модели для классификации токсичности.

Поддерживает:
- CPU: динамическая квантизация (AVX2/AVX512) — ускорение инференса на CPU
- GPU: квантизация для TensorRT — оптимизация для NVIDIA GPU

Использование:
  1. Отдельный скрипт:
     python scripts/quantize_bert_onnx.py models/bert -o models/bert_onnx --device cpu
     python scripts/quantize_bert_onnx.py models/bert -o models/bert_onnx_gpu --device gpu

  2. Импорт после обучения:
     from scripts.quantize_bert_onnx import quantize_bert_to_onnx
     quantize_bert_to_onnx("models/bert", "models/bert_onnx", device="cpu")
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

# Добавляем путь к корню проекта для импортов
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))


def _ensure_optimum_onnx():
    """Проверяет наличие optimum[onnxruntime] и возвращает модули."""
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        return ORTModelForSequenceClassification, ORTQuantizer, AutoQuantizationConfig
    except ImportError as e:
        raise ImportError(
            "Для ONNX квантизации установите: pip install optimum[onnxruntime]\n"
            "Для GPU (TensorRT): pip install optimum[onnxruntime-gpu]"
        ) from e


def _ensure_calibration_config():
    """Проверяет наличие AutoCalibrationConfig для статической квантизации."""
    try:
        from optimum.onnxruntime.configuration import AutoCalibrationConfig
        return AutoCalibrationConfig
    except ImportError:
        return None


def quantize_bert_to_onnx_cpu(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    cpu_arch: str = "avx2",
    per_channel: bool = False,
    export_only: bool = False,
) -> Path:
    """
    Экспорт и динамическая квантизация BERT модели для CPU.

    Args:
        model_path: Путь к модели HuggingFace (config.json, pytorch_model.bin)
        output_dir: Директория для сохранения ONNX модели
        cpu_arch: Архитектура CPU: avx2, avx512, avx512_vnni, arm64
        per_channel: Per-channel квантизация (точнее, но тяжелее)
        export_only: Только экспорт в ONNX без квантизации

    Returns:
        Путь к сохранённой модели
    """
    ORTModel, ORTQuantizer, AutoQuantizationConfig = _ensure_optimum_onnx()

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка модели из {model_path}...")
    ort_model = ORTModel.from_pretrained(
        str(model_path),
        export=True,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    if export_only:
        ort_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"ONNX модель сохранена в {output_dir}")
        return output_dir

    # Сохраняем ONNX во временную директорию для квантизации
    import tempfile
    arch_map = {
        "avx2": lambda: AutoQuantizationConfig.avx2(is_static=False, per_channel=per_channel),
        "avx512": lambda: AutoQuantizationConfig.avx512(is_static=False, per_channel=per_channel),
        "avx512_vnni": lambda: AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=per_channel),
        "arm64": lambda: AutoQuantizationConfig.arm64(is_static=False, per_channel=per_channel),
    }
    if cpu_arch not in arch_map:
        raise ValueError(f"Неизвестная архитектура CPU: {cpu_arch}. Доступно: {list(arch_map.keys())}")
    qconfig = arch_map[cpu_arch]()

    with tempfile.TemporaryDirectory() as tmpdir:
        ort_model.save_pretrained(tmpdir)
        quantizer = ORTQuantizer.from_pretrained(tmpdir)
        print(f"Применение динамической квантизации (CPU, {cpu_arch})...")
        quantizer.quantize(save_dir=str(output_dir), quantization_config=qconfig)

    tokenizer.save_pretrained(str(output_dir))
    import shutil
    for fname in ("config.json", "params.json"):
        src = model_path / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
    print(f"Квантизированная ONNX модель сохранена в {output_dir}")
    return output_dir


def quantize_bert_to_onnx_gpu(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    calibration_texts: Optional[List[str]] = None,
    num_calibration_samples: int = 100,
    per_channel: bool = True,
    export_only: bool = False,
) -> Path:
    """
    Экспорт и квантизация BERT модели для GPU (TensorRT).

    TensorRT использует статическую квантизацию — требуется калибровочный датасет.
    Если calibration_texts не задан, используются сгенерированные паддинг-последовательности.

    Args:
        model_path: Путь к модели HuggingFace
        output_dir: Директория для сохранения ONNX модели
        calibration_texts: Тексты для калибровки (рекомендуется из обучающей выборки)
        num_calibration_samples: Число сэмплов для калибровки, если texts не заданы
        per_channel: Per-channel квантизация
        export_only: Только экспорт в ONNX без квантизации

    Returns:
        Путь к сохранённой модели
    """
    ORTModel, ORTQuantizer, AutoQuantizationConfig = _ensure_optimum_onnx()
    AutoCalibrationConfig = _ensure_calibration_config()

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка модели из {model_path}...")
    ort_model = ORTModel.from_pretrained(
        str(model_path),
        export=True,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    if export_only:
        ort_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"ONNX модель (GPU-ready) сохранена в {output_dir}")
        return output_dir

    if AutoCalibrationConfig is None:
        raise ImportError(
            "Для TensorRT квантизации требуется AutoCalibrationConfig. "
            "Убедитесь, что установлен optimum[onnxruntime-gpu]"
        )

    # Подготовка калибровочного датасета
    max_len = min(getattr(tokenizer, "model_max_length", 512) or 512, 512)

    if calibration_texts:
        def _tokenize(t: str):
            out = tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors=None,
            )
            return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

        calib_data = [_tokenize(str(t)) for t in calibration_texts[:num_calibration_samples]]
    else:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        calib_data = [
            {"input_ids": [pad_id] * max_len, "attention_mask": [1] * max_len}
            for _ in range(num_calibration_samples)
        ]

    # HuggingFace Dataset для совместимости с AutoCalibrationConfig
    try:
        from datasets import Dataset
        calibration_dataset = Dataset.from_list(calib_data)
    except ImportError:
        class _ListDataset:
            def __init__(self, data):
                self._data = data
            def __iter__(self):
                return iter(self._data)
            def __len__(self):
                return len(self._data)
        calibration_dataset = _ListDataset(calib_data)

    # TensorRT конфиг (статическая квантизация)
    qconfig = AutoQuantizationConfig.tensorrt(per_channel=per_channel)

    # Калибровка
    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

    print("Выполнение калибровки для TensorRT...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    ranges = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
    )

    print("Применение TensorRT квантизации...")
    quantizer.quantize(
        save_dir=str(output_dir),
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
    )

    tokenizer.save_pretrained(str(output_dir))
    import shutil
    for fname in ("config.json", "params.json"):
        src = model_path / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
    print(f"Квантизированная ONNX модель (TensorRT) сохранена в {output_dir}")
    return output_dir


def quantize_bert_to_onnx(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    device: str = "cpu",
    calibration_texts: Optional[List[str]] = None,
    **kwargs,
) -> Path:
    """
    Универсальная функция квантизации BERT в ONNX.

    Args:
        model_path: Путь к модели HuggingFace
        output_dir: Директория для сохранения
        device: "cpu" или "gpu"
        calibration_texts: Тексты для калибровки (только для GPU)
        **kwargs: Доп. аргументы для quantize_bert_to_onnx_cpu или quantize_bert_to_onnx_gpu

    Returns:
        Путь к сохранённой модели
    """
    device = device.lower().strip()
    if device == "cpu":
        return quantize_bert_to_onnx_cpu(model_path, output_dir, **kwargs)
    elif device in ("gpu", "cuda", "tensorrt"):
        return quantize_bert_to_onnx_gpu(
            model_path, output_dir,
            calibration_texts=calibration_texts,
            **kwargs,
        )
    else:
        raise ValueError(f"Неизвестное устройство: {device}. Используйте 'cpu' или 'gpu'")


def main():
    parser = argparse.ArgumentParser(
        description="ONNX квантизация BERT модели для классификации токсичности"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Путь к модели HuggingFace (директория с config.json, pytorch_model.bin)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Директория для сохранения ONNX модели (по умолчанию: <model_path>_onnx_<device>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Устройство: cpu (AVX2/AVX512) или gpu (TensorRT)",
    )
    parser.add_argument(
        "--cpu-arch",
        type=str,
        choices=["avx2", "avx512", "avx512_vnni", "arm64"],
        default="avx2",
        help="Архитектура CPU (только для --device cpu)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Per-channel квантизация (точнее, но тяжелее)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Только экспорт в ONNX без квантизации",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Путь к CSV/JSON с колонкой 'text' для калибровки (только для GPU)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=100,
        help="Число сэмплов для калибровки при GPU (если --calibration-data не задан)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        parser.error(f"Модель не найдена: {model_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"{model_path}_onnx_{args.device}"
    output_dir = Path(output_dir)

    calibration_texts = None
    if args.calibration_data:
        calib_path = Path(args.calibration_data)
        if not calib_path.exists():
            parser.error(f"Файл калибровки не найден: {calib_path}")
        import pandas as pd
        df = pd.read_csv(calib_path) if calib_path.suffix == ".csv" else pd.read_json(calib_path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        calibration_texts = df[text_col].astype(str).dropna().tolist()

    if args.device == "cpu":
        quantize_bert_to_onnx_cpu(
            model_path,
            output_dir,
            cpu_arch=args.cpu_arch,
            per_channel=args.per_channel,
            export_only=args.export_only,
        )
    else:
        quantize_bert_to_onnx_gpu(
            model_path,
            output_dir,
            calibration_texts=calibration_texts,
            num_calibration_samples=args.num_calibration_samples,
            per_channel=args.per_channel,
            export_only=args.export_only,
        )

    print("Готово.")


if __name__ == "__main__":
    main()
