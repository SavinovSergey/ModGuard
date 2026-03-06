"""Тест функции _convert_to_json_serializable"""
import json
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scripts.toxicity.train_rnn import RNNModelTrainer

def test_convert_to_json_serializable():
    """Тестирование функции _convert_to_json_serializable"""
    
    print("Тестирование функции _convert_to_json_serializable\n")

    eps = 1e-7
    
    # Тест 1: numpy float32
    print("Тест 1: numpy.float32")
    value = np.float32(0.95)
    result = RNNModelTrainer._convert_to_json_serializable(value)
    print(f"  Вход: {value} (тип: {type(value)})")
    print(f"  Выход: {result} (тип: {type(result)})")
    assert isinstance(result, float), f"Ожидался float, получен {type(result)}"
    assert abs(result - 0.95) < eps, f"Ожидалось 0.95, получено {result}"
    print("  ✓ Успешно\n")
    
    # Тест 2: numpy float64
    print("Тест 2: numpy.float64")
    value = np.float64(0.123456789)
    result = RNNModelTrainer._convert_to_json_serializable(value)
    print(f"  Вход: {value} (тип: {type(value)})")
    print(f"  Выход: {result} (тип: {type(result)})")
    assert isinstance(result, float), f"Ожидался float, получен {type(result)}"
    print("  ✓ Успешно\n")
    
    # Тест 3: numpy int32
    print("Тест 3: numpy.int32")
    value = np.int32(42)
    result = RNNModelTrainer._convert_to_json_serializable(value)
    print(f"  Вход: {value} (тип: {type(value)})")
    print(f"  Выход: {result} (тип: {type(result)})")
    assert isinstance(result, int), f"Ожидался int, получен {type(result)}"
    assert abs(result - 42) < eps, f"Ожидалось 42, получено {result}"
    print("  ✓ Успешно\n")
    
    # Тест 4: numpy int64
    print("Тест 4: numpy.int64")
    value = np.int64(100)
    result = RNNModelTrainer._convert_to_json_serializable(value)
    print(f"  Вход: {value} (тип: {type(value)})")
    print(f"  Выход: {result} (тип: {type(result)})")
    assert isinstance(result, int), f"Ожидался int, получен {type(result)}"
    print("  ✓ Успешно\n")
    
    # Тест 5: словарь с numpy значениями
    print("Тест 5: словарь с numpy значениями")
    metadata = {
        'best_score': np.float32(0.95),
        'optimal_threshold': np.float64(0.87),
        'random_state': np.int32(42),
        'dropout': 0.2,  # обычный float
        'use_cv': True,  # bool
        'tokenizer_type': 'bpe'  # str
    }
    result = RNNModelTrainer._convert_to_json_serializable(metadata)
    print(f"  Вход: {metadata}")
    print(f"  Выход: {result}")
    assert isinstance(result['best_score'], float), "best_score должен быть float"
    assert isinstance(result['optimal_threshold'], float), "optimal_threshold должен быть float"
    assert isinstance(result['random_state'], int), "random_state должен быть int"
    assert isinstance(result['dropout'], float), "dropout должен быть float"
    assert isinstance(result['use_cv'], bool), "use_cv должен быть bool"
    assert isinstance(result['tokenizer_type'], str), "tokenizer_type должен быть str"
    print("  ✓ Успешно\n")
    
    # Тест 6: вложенный словарь
    print("Тест 6: вложенный словарь")
    metadata = {
        'params': {
            'focal_gamma': np.float32(2.0),
            'focal_alpha': np.float64(0.5)
        },
        'nested': {
            'deep': {
                'value': np.int32(100)
            }
        }
    }
    result = RNNModelTrainer._convert_to_json_serializable(metadata)
    print(f"  Вход: {metadata}")
    print(f"  Выход: {result}")
    assert isinstance(result['params']['focal_gamma'], float)
    assert isinstance(result['params']['focal_alpha'], float)
    assert isinstance(result['nested']['deep']['value'], int)
    print("  ✓ Успешно\n")
    
    # Тест 7: список с numpy значениями
    print("Тест 7: список с numpy значениями")
    values = [np.float32(0.1), np.float64(0.2), np.int32(3), 'string', True]
    result = RNNModelTrainer._convert_to_json_serializable(values)
    print(f"  Вход: {values}")
    print(f"  Выход: {result}")
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert isinstance(result[2], int)
    assert isinstance(result[3], str)
    assert isinstance(result[4], bool)
    print("  ✓ Успешно\n")
    
    # Тест 8: None значения
    print("Тест 8: None значения")
    metadata = {
        'value': np.float32(0.5),
        'none_value': None,
        'cv_folds': None
    }
    result = RNNModelTrainer._convert_to_json_serializable(metadata)
    print(f"  Вход: {metadata}")
    print(f"  Выход: {result}")
    assert result['none_value'] is None
    assert result['cv_folds'] is None
    print("  ✓ Успешно\n")
    
    # Тест 9: реальный пример из train_rnn.py
    print("Тест 9: реальный пример метаданных из train_rnn.py")
    metadata = {
        'tokenizer_type': 'bpe',
        'rnn_type': 'gru',
        'embedding_dim': 100,
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.2,
        'bidirectional': False,
        'max_length': 512,
        'vocab_size': 20000,
        'best_score': np.float32(0.95),  # Это может быть numpy тип
        'best_score_metric': 'average_precision',
        'optimal_threshold': np.float64(0.87),  # Это может быть numpy тип
        'random_state': 42,
        'loss_type': 'bce',
        'focal_gamma': 2.0,
        'focal_alpha': None,
        'focal_auto_alpha': True,
        'embedding_dropout': 0.0,
        'weight_decay': 1e-4,
        'use_cv': True,
        'cv_folds': 4,
        'use_layer_norm': False,
        'max_grad_norm': 1.0,
        'use_lr_schedule': True,
        'lr_schedule_patience': 2,
        'lr_schedule_factor': 0.5
    }
    result = RNNModelTrainer._convert_to_json_serializable(metadata)
    print(f"  Преобразовано {len(metadata)} полей")
    
    # Попытка сериализации в JSON
    try:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        print(f"  JSON сериализация успешна (длина: {len(json_str)} символов)")
        print("  ✓ Успешно\n")
    except Exception as e:
        print(f"  ✗ Ошибка JSON сериализации: {e}")
        raise
    
    # Тест 10: numpy array
    print("Тест 10: numpy array")
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    result = RNNModelTrainer._convert_to_json_serializable(arr)
    print(f"  Вход: {arr} (тип: {type(arr)})")
    print(f"  Выход: {result} (тип: {type(result)})")
    assert isinstance(result, list), "Результат должен быть списком"
    assert len(result) == 5, "Длина должна быть 5"
    print("  ✓ Успешно\n")
    
    print("=" * 60)
    print("Все тесты пройдены успешно! ✓")
    print("=" * 60)

if __name__ == '__main__':
    test_convert_to_json_serializable()

