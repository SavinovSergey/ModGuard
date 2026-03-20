"""
Скрипт ONNX квантизации BERT модели для классификации токсичности.

Поддерживает:
- CPU: динамическая квантизация (AVX2/AVX512) — ускорение инференса на CPU
- GPU: квантизация для TensorRT — оптимизация для NVIDIA GPU

Использование:
  1. Отдельный скрипт:
     python scripts/toxicity/quantize_bert_onnx.py models/toxicity/bert -o models/toxicity/bert/onnx --device cpu
     python scripts/toxicity/quantize_bert_onnx.py models/toxicity/bert -o models/toxicity/bert/onnx_gpu --device gpu

  2. Импорт после обучения:
     from scripts.toxicity.quantize_bert_onnx import quantize_bert_to_onnx
     quantize_bert_to_onnx("models/toxicity/bert", "models/toxicity/bert/onnx", device="cpu")
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

# Корень проекта (скрипт в scripts/toxicity/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))


def _load_state_dict_from_dir(model_dir: Path):
    """Загружает state_dict из директории (pytorch_model.bin или model.safetensors)."""
    if (model_dir / "model.safetensors").exists():
        from safetensors.torch import load_file
        return load_file(model_dir / "model.safetensors")
    if (model_dir / "pytorch_model.bin").exists():
        import torch
        return torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    raise FileNotFoundError(
        f"Не найден pytorch_model.bin или model.safetensors в {model_dir}"
    )


def _prepare_model_dir_for_export(model_path: Path):
    """
    Подготавливает директорию с моделью для экспорта в ONNX.
    Если чекпоинт дообученный (голова classifier.1.weight/bias), переименовывает ключи
    в classifier.weight/bias, чтобы стандартный BertForSequenceClassification загрузил веса.
    Возвращает путь к временной директории (caller должен удалить после использования).
    """
    model_path = Path(model_path)
    state_dict = _load_state_dict_from_dir(model_path)

    if "classifier.1.weight" not in state_dict:
        # Стандартная HF-модель — можно экспортировать из исходной папки
        return None

    # Дообученная модель: голова Sequential(Dropout, Linear) → ключи classifier.1.*
    # Приводим к виду classifier.weight / classifier.bias для HF
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "classifier.1.weight":
            new_state_dict["classifier.weight"] = v
        elif k == "classifier.1.bias":
            new_state_dict["classifier.bias"] = v
        else:
            new_state_dict[k] = v

    tmpdir = tempfile.mkdtemp(prefix="bert_onnx_export_")
    tmpdir_path = Path(tmpdir)
    for f in model_path.iterdir():
        if f.is_file():
            shutil.copy2(f, tmpdir_path / f.name)
    # Конфиг: одна выходная логит → num_labels=1
    import json
    config_path = tmpdir_path / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["num_labels"] = 1
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    # Перезаписываем веса переименованным state_dict
    import torch
    torch.save(new_state_dict, tmpdir_path / "pytorch_model.bin")
    if (tmpdir_path / "model.safetensors").exists():
        (tmpdir_path / "model.safetensors").unlink()
    return tmpdir_path


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

    model_source = str(model_path)
    model_dir: Optional[Path] = None
    try:
        cand = Path(model_source)
        if cand.exists():
            model_dir = cand
    except Exception:
        model_dir = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_dir = None
    load_path = model_source
    if model_dir is not None:
        export_dir = _prepare_model_dir_for_export(model_dir)
        load_path = str(export_dir if export_dir is not None else model_dir)
        if export_dir is not None:
            print("Подготовлена дообученная модель (голова classifier.1 → classifier) для экспорта.")

    print(f"Загрузка модели из {model_source}...")
    try:
        ort_model = ORTModel.from_pretrained(
            load_path,
            export=True,
        )
    finally:
        if export_dir is not None:
            shutil.rmtree(export_dir, ignore_errors=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    if export_only:
        ort_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"ONNX модель сохранена в {output_dir}")
        return output_dir

    # Сохраняем ONNX во временную директорию для квантизации
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
    if model_dir is not None:
        for fname in ("config.json", "params.json"):
            src = model_dir / fname
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

    model_source = str(model_path)
    model_dir: Optional[Path] = None
    try:
        cand = Path(model_source)
        if cand.exists():
            model_dir = cand
    except Exception:
        model_dir = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_dir = None
    load_path = model_source
    if model_dir is not None:
        export_dir = _prepare_model_dir_for_export(model_dir)
        load_path = str(export_dir if export_dir is not None else model_dir)
        if export_dir is not None:
            print("Подготовлена дообученная модель (голова classifier.1 → classifier) для экспорта.")

    print(f"Загрузка модели из {model_source}...")
    try:
        ort_model = ORTModel.from_pretrained(
            load_path,
            export=True,
        )
    finally:
        if export_dir is not None:
            shutil.rmtree(export_dir, ignore_errors=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source)

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
        help="Директория для сохранения ONNX модели (по умолчанию: <model_path>/onnx_<device>)",
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

    model_path_input = args.model_path
    model_dir = None
    try:
        cand = Path(model_path_input)
        if cand.exists():
            model_dir = cand
    except Exception:
        model_dir = None

    # model_path_input может быть как локальной директорией, так и HF model-id.
    is_local = model_dir is not None

    output_dir = args.output_dir
    if output_dir is None:
        if is_local:
            output_dir = f"{model_dir}/onnx_{args.device}"
        else:
            # Нейминг для HF model-id: SergeySavinov/rubert-tiny-toxicity -> SergeySavinov_rubert-tiny-toxicity
            safe_id = model_path_input.replace("/", "_")
            output_dir = f"{safe_id}/onnx_{args.device}"
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
            model_path_input,
            output_dir,
            cpu_arch=args.cpu_arch,
            per_channel=args.per_channel,
            export_only=args.export_only,
        )
    else:
        quantize_bert_to_onnx_gpu(
            model_path_input,
            output_dir,
            calibration_texts=calibration_texts,
            num_calibration_samples=args.num_calibration_samples,
            per_channel=args.per_channel,
            export_only=args.export_only,
        )

    print("Готово.")


if __name__ == "__main__":
    main()
