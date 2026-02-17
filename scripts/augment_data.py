"""Скрипт для предварительной аугментации данных"""
import argparse
from pathlib import Path
import pandas as pd
import os
import sys

# Добавляем путь к app для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.augmentation import DataAugmenter


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description='Аугментация данных для обучения моделей классификации токсичности'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Путь к входному файлу с данными (parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Путь к выходному файлу для сохранения аугментированных данных (parquet)'
    )
    parser.add_argument(
        '--text-col',
        type=str,
        default='text',
        help='Название колонки с текстом (по умолчанию: text)'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='label',
        help='Название колонки с метками (по умолчанию: label)'
    )
    parser.add_argument(
        '--no-char-noise',
        action='store_true',
        help='Отключить шум на уровне символов'
    )
    parser.add_argument(
        '--char-noise-intensity',
        type=float,
        default=0.04,
        help='Интенсивность шума (процент символов для изменения, по умолчанию: 0.04 = 4%%)'
    )
    parser.add_argument(
        '--char-noise-prob',
        type=float,
        default=0.5,
        help='Вероятность применения шума к примеру (по умолчанию: 0.5)'
    )
    parser.add_argument(
        '--toxic-noise-multiplier',
        type=float,
        default=1.5,
        help='Множитель интенсивности шума для токсичных примеров (по умолчанию: 1.5)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed для воспроизводимости (по умолчанию: 42)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Размер чанка для обработки больших файлов (None = обработать весь файл сразу)'
    )
    parser.add_argument(
        '--apply-back-translation',
        action='store_true',
        help='Применить двойной перевод'
    )
    parser.add_argument(
        '--translation-device',
        type=str,
        default=None,
        help='Устройство для перевода (cuda:0, cuda:1, cpu или None для автоопределения)'
    )
    parser.add_argument(
        '--translation-batch-size',
        type=int,
        default=32,
        help='Размер батча для перевода (по умолчанию: 32)'
    )
    parser.add_argument(
        '--translation-max-new-tokens',
        type=int,
        default=100,
        help='Максимальная количество новых токенов при переводе(по умолчанию: 100)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=None,
        help='Начальный индекс строк для обработки (для параллельной обработки)'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='Конечный индекс строк для обработки (для параллельной обработки)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='ID GPU для использования (0 или 1). Если указан, автоматически устанавливает translation-device=cuda:{gpu_id}'
    )
    
    args = parser.parse_args()
    
    # Если указан gpu_id, устанавливаем translation_device
    if args.gpu_id is not None:
        args.translation_device = f'cuda:{args.gpu_id}'
        print(f"Использование GPU {args.gpu_id}: {args.translation_device}")
    
    # Проверка входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: файл {input_path} не найден")
        return 1
    
    # Создание директории для выходного файла
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка данных из {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Загружено {len(df)} примеров")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Проверка наличия необходимых колонок
    if args.text_col not in df.columns:
        print(f"Ошибка: колонка '{args.text_col}' не найдена в данных")
        return 1
    
    if args.label_col not in df.columns:
        print(f"Ошибка: колонка '{args.label_col}' не найдена в данных")
        return 1
    
    # Статистика по классам
    print(f"\nРаспределение классов:")
    class_counts = df[args.label_col].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  Класс {label}: {count} примеров")
    
    # Фильтрация данных по индексам, если указаны
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(df)
        print(f"\nФильтрация данных: индексы {start_idx} - {end_idx}")
        df = df.iloc[start_idx:end_idx].copy()
        print(f"Осталось {len(df)} примеров для обработки")
    
    # Создание аугментера
    print(f"\nИнициализация аугментера...")
    print(f"  Двойной перевод: {'включен' if args.apply_back_translation else 'отключен'}")
    if args.apply_back_translation:
        print(f"    Устройство: {args.translation_device or 'автоопределение'}")
        print(f"    Размер батча: {args.translation_batch_size}")
        print(f"    Максимальная длина генерации: {args.translation_max_new_tokens}")
    print(f"  Шум на уровне символов: {'включен' if not args.no_char_noise else 'отключен'}")
    if not args.no_char_noise:
        print(f"  Интенсивность шума: {args.char_noise_intensity * 100:.1f}%")
        print(f"  Вероятность применения шума: {args.char_noise_prob * 100:.1f}%")
        print(f"  Множитель для токсичных: {args.toxic_noise_multiplier}x")
    
    try:
        augmenter = DataAugmenter(
            apply_back_translation=args.apply_back_translation,
            apply_char_noise=not args.no_char_noise,
            char_noise_intensity=args.char_noise_intensity,
            char_noise_prob=args.char_noise_prob,
            toxic_noise_multiplier=args.toxic_noise_multiplier,
            translation_device=args.translation_device,
            translation_batch_size=args.translation_batch_size,
            translation_max_new_tokens=args.translation_max_new_tokens,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"Ошибка при инициализации аугментера: {e}")
        return 1
    
    # Применение аугментации
    if args.chunk_size:
        # Обработка по частям для больших файлов
        print(f"\nОбработка данных по частям (размер чанка: {args.chunk_size})...")
        all_augmented_dfs = []
        
        for i in range(0, len(df), args.chunk_size):
            chunk = df.iloc[i:i + args.chunk_size]
            print(f"\nОбработка чанка {i // args.chunk_size + 1} (строки {i}-{min(i + args.chunk_size, len(df))})...")
            augmented_chunk = augmenter.augment_dataframe(
                chunk,
                text_col=args.text_col,
                label_col=args.label_col
            )
            all_augmented_dfs.append(augmented_chunk)
        
        df_augmented = pd.concat(all_augmented_dfs, ignore_index=True)
    else:
        # Обработка всего файла сразу
        print(f"\nПрименение аугментации ко всем данным...")
        df_augmented = augmenter.augment_dataframe(
            df,
            text_col=args.text_col,
            label_col=args.label_col
        )
    
    # Сохранение результата
    print(f"\nСохранение результата в {output_path}...")
    df_augmented.to_parquet(output_path, index=False)
    print(f"Сохранено {len(df_augmented)} примеров")
    
    # Финальная статистика
    print(f"\nФинальная статистика:")
    print(f"  Оригинальных примеров: {len(df)}")
    print(f"  Аугментированных примеров: {len(df_augmented) - len(df)}")
    print(f"  Всего примеров: {len(df_augmented)}")
    
    if 'augmentation_type' in df_augmented.columns:
        print(f"\nРаспределение по типам аугментации:")
        aug_stats = df_augmented['augmentation_type'].value_counts()
        for aug_type, count in aug_stats.items():
            percentage = (count / len(df_augmented)) * 100
            print(f"  {aug_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nРаспределение классов после аугментации:")
    class_counts = df_augmented[args.label_col].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  Класс {label}: {count} примеров")
    
    print("\nАугментация завершена успешно!")
    return 0


if __name__ == '__main__':
    exit(main())

