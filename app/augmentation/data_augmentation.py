"""Модуль для аугментации текстовых данных"""
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    MarianMTModel = None
    MarianTokenizer = None


class CharNoiseAugmenter:
    """Добавление шума на уровне символов"""
    
    # QWERTY раскладка для русской клавиатуры
    QWERTY_RU = {
        'й': ['ц', 'ф'],
        'ц': ['й', 'у', 'ы'],
        'у': ['ц', 'к', 'е'],
        'к': ['у', 'е', 'н'],
        'е': ['к', 'н', 'г'],
        'н': ['е', 'г', 'ш'],
        'г': ['н', 'ш', 'щ'],
        'ш': ['г', 'щ', 'з'],
        'щ': ['ш', 'з', 'х'],
        'з': ['щ', 'х', 'ъ'],
        'х': ['з', 'ъ'],
        'ъ': ['х'],
        'ф': ['й', 'ы', 'в'],
        'ы': ['ф', 'в', 'а'],
        'в': ['ы', 'а', 'п'],
        'а': ['в', 'п', 'р'],
        'п': ['а', 'р', 'о'],
        'р': ['п', 'о', 'л'],
        'о': ['р', 'л', 'д'],
        'л': ['о', 'д', 'ж'],
        'д': ['л', 'ж', 'э'],
        'ж': ['д', 'э'],
        'э': ['ж'],
        'я': ['ч', 'с', 'м'],
        'ч': ['я', 'с', 'м'],
        'с': ['ч', 'м', 'и'],
        'м': ['с', 'и', 'т'],
        'и': ['м', 'т', 'ь'],
        'т': ['и', 'ь', 'б'],
        'ь': ['т', 'б', 'ю'],
        'б': ['ь', 'ю'],
        'ю': ['б']
    }
    
    # LeetSpeak замены для русского языка
    LEET_DICT = {
        'а': ['4', '@'],
        'о': ['0', 'о'],
        'е': ['3', 'ё'],
        'и': ['1', '|'],
        'с': ['$', 'с'],
        'з': ['3', 'з'],
        'т': ['7', 'т'],
        'б': ['6', 'б'],
        'г': ['г'],
        'д': ['д'],
        'к': ['к'],
        'л': ['л'],
        'м': ['м'],
        'н': ['н'],
        'п': ['п'],
        'р': ['р'],
        'у': ['у'],
        'ф': ['ф'],
        'х': ['х'],
        'ц': ['ц'],
        'ч': ['ч'],
        'ш': ['ш'],
        'щ': ['щ'],
        'ъ': ['ъ'],
        'ы': ['ы'],
        'ь': ['ь'],
        'э': ['э'],
        'ю': ['ю'],
        'я': ['я']
    }
    
    def __init__(
        self,
        noise_intensity: float = 0.04,
        qwerty_prob: float = 0.25,
        insert_prob: float = 0.25,
        delete_prob: float = 0.25,
        replace_prob: float = 0.25,
        leet_prob: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Args:
            noise_intensity: Процент символов для изменения (0.0-1.0)
            qwerty_prob: Вероятность применения QWERTY замены
            insert_prob: Вероятность добавления символа
            delete_prob: Вероятность удаления символа
            replace_prob: Вероятность замены символа
            leet_prob: Вероятность LeetSpeak замены
            random_state: Seed для воспроизводимости
        """
        self.noise_intensity = noise_intensity
        self.qwerty_prob = qwerty_prob
        self.insert_prob = insert_prob
        self.delete_prob = delete_prob
        self.replace_prob = replace_prob
        self.leet_prob = leet_prob
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _qwerty_replace(self, text: str) -> str:
        """Замена символов на соседние клавиши QWERTY"""
        result = list(text)
        num_changes = max(1, int(len(text) * self.noise_intensity))
        changed_indices = set()
        
        for _ in range(num_changes):
            idx = random.randint(0, len(result) - 1)
            if idx in changed_indices:
                continue
            
            char = result[idx].lower()
            if char in self.QWERTY_RU:
                neighbors = self.QWERTY_RU[char]
                replacement = random.choice(neighbors)
                # Сохраняем регистр
                if result[idx].isupper():
                    replacement = replacement.upper()
                result[idx] = replacement
                changed_indices.add(idx)
        
        return ''.join(result)
    
    def _insert_char(self, text: str) -> str:
        """Случайная вставка символов"""
        result = list(text)
        num_insertions = max(1, int(len(text) * self.noise_intensity))
        
        # Русский алфавит для вставки
        russian_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        
        for _ in range(num_insertions):
            idx = random.randint(0, len(result))
            char_to_insert = random.choice(russian_chars)
            result.insert(idx, char_to_insert)
        
        return ''.join(result)
    
    def _delete_char(self, text: str) -> str:
        """Случайное удаление символов"""
        if len(text) <= 1:
            return text
        
        result = list(text)
        num_deletions = max(1, min(int(len(text) * self.noise_intensity), len(text) - 1))
        
        # Удаляем символы в случайных позициях
        indices_to_delete = sorted(random.sample(range(len(result)), num_deletions), reverse=True)
        for idx in indices_to_delete:
            result.pop(idx)
        
        return ''.join(result)
    
    def _replace_char(self, text: str) -> str:
        """Случайная замена символов"""
        result = list(text)
        num_replacements = max(1, int(len(text) * self.noise_intensity))
        changed_indices = set()
        
        # Русский алфавит для замены
        russian_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        
        for _ in range(num_replacements):
            idx = random.randint(0, len(result) - 1)
            if idx in changed_indices:
                continue
            
            char = result[idx].lower()
            if char.isalpha() and char in russian_chars:
                replacement = random.choice(russian_chars)
                # Сохраняем регистр
                if result[idx].isupper():
                    replacement = replacement.upper()
                result[idx] = replacement
                changed_indices.add(idx)
        
        return ''.join(result)
    
    def _leet_speak(self, text: str) -> str:
        """LeetSpeak замена"""
        result = list(text)
        num_replacements = max(1, int(len(text) * self.noise_intensity))
        changed_indices = set()
        
        for _ in range(num_replacements):
            idx = random.randint(0, len(result) - 1)
            if idx in changed_indices:
                continue
            
            char = result[idx].lower()
            if char in self.LEET_DICT:
                replacement = random.choice(self.LEET_DICT[char])
                # Сохраняем регистр
                if result[idx].isupper():
                    replacement = replacement.upper()
                result[idx] = replacement
                changed_indices.add(idx)
        
        return ''.join(result)
    
    def _combined_noise(self, text: str) -> str:
        """Комбинированный шум - применение нескольких типов"""
        result = text
        
        # Применяем несколько типов шума последовательно с меньшей интенсивностью
        noise_types = []
        if random.random() < self.qwerty_prob:
            noise_types.append('qwerty')
        if random.random() < self.insert_prob:
            noise_types.append('insert')
        if random.random() < self.delete_prob:
            noise_types.append('delete')
        if random.random() < self.replace_prob:
            noise_types.append('replace')
        if random.random() < self.leet_prob:
            noise_types.append('leet')
        
        # Если нет выбранных типов, выбираем случайный
        if not noise_types:
            noise_types = [random.choice(['qwerty', 'insert', 'delete', 'replace', 'leet'])]
        
        # Применяем каждый тип с уменьшенной интенсивностью
        local_intensity = self.noise_intensity / len(noise_types)
        original_intensity = self.noise_intensity
        self.noise_intensity = local_intensity
        
        for noise_type in noise_types[:2]:  # Максимум 2 типа за раз
            if noise_type == 'qwerty':
                result = self._qwerty_replace(result)
            elif noise_type == 'insert':
                result = self._insert_char(result)
            elif noise_type == 'delete':
                result = self._delete_char(result)
            elif noise_type == 'replace':
                result = self._replace_char(result)
            elif noise_type == 'leet':
                result = self._leet_speak(result)
        
        self.noise_intensity = original_intensity
        return result
    
    def augment(
        self,
        text: str,
        noise_type: Optional[str] = None,
        preserve_toxicity: bool = True
    ) -> Tuple[str, str]:
        """
        Применяет шум к тексту
        
        Args:
            text: Исходный текст
            noise_type: Тип шума ('qwerty', 'insert', 'delete', 'replace', 'leet', 'combined')
                       Если None, выбирается случайно
            preserve_toxicity: Сохранять ли токсичность (не реализовано пока)
        
        Returns:
            Tuple[augmented_text, augmentation_type]
        """
        if not text or len(text.strip()) == 0:
            return text, 'original'
        
        if noise_type is None:
            # Выбираем тип шума на основе вероятностей
            rand = random.random()
            if rand < self.qwerty_prob:
                noise_type = 'qwerty'
            elif rand < self.qwerty_prob + self.insert_prob:
                noise_type = 'insert'
            elif rand < self.qwerty_prob + self.insert_prob + self.delete_prob:
                noise_type = 'delete'
            elif rand < self.qwerty_prob + self.insert_prob + self.delete_prob + self.replace_prob:
                noise_type = 'replace'
            elif rand < self.qwerty_prob + self.insert_prob + self.delete_prob + self.replace_prob + self.leet_prob:
                noise_type = 'leet'
            else:
                noise_type = 'combined'
        
        if noise_type == 'qwerty':
            augmented = self._qwerty_replace(text)
            aug_type = 'char_noise_qwerty'
        elif noise_type == 'insert':
            augmented = self._insert_char(text)
            aug_type = 'char_noise_insert'
        elif noise_type == 'delete':
            augmented = self._delete_char(text)
            aug_type = 'char_noise_delete'
        elif noise_type == 'replace':
            augmented = self._replace_char(text)
            aug_type = 'char_noise_replace'
        elif noise_type == 'leet':
            augmented = self._leet_speak(text)
            aug_type = 'char_noise_leet'
        elif noise_type == 'combined':
            augmented = self._combined_noise(text)
            aug_type = 'char_noise_combined'
        else:
            augmented = text
            aug_type = 'original'
        
        return augmented, aug_type


class BackTranslationAugmenter:
    """Двойной перевод через модели HuggingFace (Helsinki-NLP OPUS MT)"""
    
    def __init__(
        self,
        model_ru_en: str = "Helsinki-NLP/opus-mt-ru-en",
        model_en_ru: str = "Helsinki-NLP/opus-mt-en-ru",
        device: Optional[str] = None,
        max_new_tokens: int = 100,
        batch_size: int = 64
    ):
        """
        Args:
            model_ru_en: Название модели для перевода RU→EN
            model_en_ru: Название модели для перевода EN→RU
            device: Устройство для вычислений ('cuda:0', 'cuda:1', 'cpu' или None для автоопределения)
            max_new_tokens: Максимальное количество новых токенов текста при переводе
            batch_size: Размер батча для перевода
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers не установлен. Установите его: pip install transformers torch"
            )
        
        # Определяем устройство
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        print(f"Загрузка моделей перевода на устройство: {self.device}")
        
        # Загружаем обе модели одновременно
        print(f"Загрузка модели RU→EN: {model_ru_en}")
        self.tokenizer_ru_en = MarianTokenizer.from_pretrained(model_ru_en)
        self.model_ru_en = MarianMTModel.from_pretrained(model_ru_en)
        self.model_ru_en = self.model_ru_en.half()  # FP16
        self.model_ru_en = self.model_ru_en.to(self.device)
        self.model_ru_en.eval()
        
        print(f"Загрузка модели EN→RU: {model_en_ru}")
        self.tokenizer_en_ru = MarianTokenizer.from_pretrained(model_en_ru)
        self.model_en_ru = MarianMTModel.from_pretrained(model_en_ru)
        self.model_en_ru = self.model_en_ru.half()  # FP16
        # self.model_en_ru = self.model_en_ru.to(self.device)   # Загрузим на gpu позже
        self.model_en_ru.eval()
        
        print("Модели загружены и готовы к работе")
    
    def _translate_batch(
        self, 
        texts: List[str], 
        tokenizer: MarianTokenizer, 
        model: MarianMTModel
    ) -> List[str]:
        """Переводит батч текстов"""
        if not texts:
            return []
        
        # Токенизация
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Максимальная длина для токенизации
        )
        
        # Перемещаем на устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Перевод с отключенными градиентами
        with torch.inference_mode():
            if self.device.startswith('cuda'):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    translated = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=2,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,
                        do_sample=False
                    )
            else:
                translated = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    do_sample=False
                )
        
        # Перемещаем обратно на CPU
        translated = translated.cpu()
        inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Декодирование
        translated_texts = tokenizer.batch_decode(
            translated, 
            skip_special_tokens=True
        )
        
        return translated_texts
    
    def augment_batch(self, texts: List[str]) -> List[Tuple[str, str]]:
        """
        Выполняет батчевый двойной перевод: RU → EN → RU
        
        Args:
            texts: Список исходных текстов на русском
        
        Returns:
            List[Tuple[augmented_text, augmentation_type]]
        """
        if not texts:
            return []
        

        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and len(str(text).strip()) > 0:
                valid_texts.append(str(text))
                valid_indices.append(i)
        
        # Инициализируем результаты оригинальными текстами
        results = [(str(text) if text else '', 'original') for text in texts]
        
        if not valid_texts:
            return results
        
        try:
            # Первый этап: RU → EN
            print(f"Перевод RU→EN для {len(valid_texts)} текстов...")
            english_texts = []
            
            for i in tqdm(range(0, len(valid_texts), self.batch_size), desc="RU→EN"):
                batch = valid_texts[i:i + self.batch_size]
                batch_translated = self._translate_batch(
                    batch, 
                    self.tokenizer_ru_en, 
                    self.model_ru_en
                )
                english_texts.extend(batch_translated)

            print("Выгрузка модели ru_en на CPU и загрузка модели en_ru на GPU.")
            self.model_ru_en = self.model_ru_en.cpu()
            self.model_en_ru = self.model_en_ru.to(self.device)

            # Второй этап: EN → RU
            print(f"Перевод EN→RU для {len(english_texts)} текстов...")
            back_translated_texts = []
            
            for i in tqdm(range(0, len(english_texts), self.batch_size), desc="EN→RU"):
                batch = english_texts[i:i + self.batch_size]
                batch_translated = self._translate_batch(
                    batch,
                    self.tokenizer_en_ru,
                    self.model_en_ru
                )
                back_translated_texts.extend(batch_translated)
            
            # Формируем результаты для валидных текстов
            for i, translated in enumerate(back_translated_texts):
                if i < len(valid_indices):
                    original_text = valid_texts[i]
                    if translated and translated != original_text:
                        results[valid_indices[i]] = (translated, 'back_translation')
        
        except Exception as e:
            print(f"Предупреждение: ошибка при переводе: {e}")
            import traceback
            traceback.print_exc()
        
        return results


class DataAugmenter:
    """Главный класс для координации всех типов аугментации"""
    
    def __init__(
        self,
        apply_back_translation: bool = True,
        apply_char_noise: bool = True,
        char_noise_intensity: float = 0.04,
        char_noise_prob: float = 0.5,
        toxic_noise_multiplier: float = 2,  # Больше шума для токсичных
        translation_device: Optional[str] = None,
        translation_batch_size: int = 32,
        translation_max_new_tokens: int = 100,
        random_state: Optional[int] = None,
        max_text_length: int = 500          # Максимальная длина аугментированного комментария
    ):
        """
        Args:
            apply_back_translation: Применять ли двойной перевод
            apply_char_noise: Применять ли шум на уровне символов
            char_noise_intensity: Интенсивность шума (процент символов)
            char_noise_prob: Вероятность применения шума к примеру
            toxic_noise_multiplier: Множитель интенсивности шума для токсичных примеров
            translation_device: Устройство для перевода ('cuda:0', 'cuda:1', 'cpu' или None)
            translation_batch_size: Размер батча для перевода
            translation_max_new_tokens: Максимальное количество новых токенов при переводе
            random_state: Seed для воспроизводимости
            max_text_length: Максимальная длина текста для обработки (символов)
        """
        self.apply_back_translation = apply_back_translation
        self.apply_char_noise = apply_char_noise
        self.char_noise_intensity = char_noise_intensity
        self.char_noise_prob = char_noise_prob
        self.toxic_noise_multiplier = toxic_noise_multiplier
        self.random_state = random_state
        self.max_text_length = max_text_length
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Инициализация аугментеров
        self.back_translator = None
        if apply_back_translation:
            try:
                # Используем модели HuggingFace для перевода
                self.back_translator = BackTranslationAugmenter(
                    device=translation_device,
                    batch_size=translation_batch_size,
                    max_new_tokens=translation_max_new_tokens
                )
            except Exception as e:
                print(f"Предупреждение: двойной перевод отключен: {e}")
                self.apply_back_translation = False
        
        self.char_noise_augmenter = None
        if apply_char_noise:
            self.char_noise_augmenter = CharNoiseAugmenter(
                noise_intensity=char_noise_intensity,
                random_state=random_state
            )
    
    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'label'
    ) -> pd.DataFrame:
        """
        Применяет аугментацию к DataFrame
        
        Args:
            df: DataFrame с колонками text и label
            text_col: Название колонки с текстом
            label_col: Название колонки с метками
        
        Returns:
            DataFrame с добавленными аугментированными примерами и колонкой augmentation_type
        """
        # Добавляем колонку augmentation_type для оригинальных примеров
        df_result = df.copy()
        df_result['augmentation_type'] = 'original'
        
        augmented_rows = []
        
        print(f"Начало аугментации {len(df)} примеров...")
        
        # Двойной перевод
        if self.apply_back_translation and self.back_translator:
            print("Применение двойного перевода...")
            
            # Подготавливаем тексты для батчевого перевода
            texts_to_translate = []
            rows_to_translate = []
            
            for idx, row in df.iterrows():
                text = row[text_col]
                if text and len(str(text).strip()) > 0 and len(str(text)) <= self.max_text_length:
                    texts_to_translate.append(str(text))
                    rows_to_translate.append((idx, row))
            
            if texts_to_translate:
                print(f"Перевод {len(texts_to_translate)} текстов (отфильтровано {len(df) - len(texts_to_translate)} текстов длиной > {self.max_text_length} символов)...")
                translation_results = self.back_translator.augment_batch(texts_to_translate)
                
                # Обрабатываем результаты
                for (idx, row), (augmented_text, aug_type) in zip(rows_to_translate, translation_results):
                    if augmented_text != row[text_col] and aug_type == 'back_translation':
                        new_row = row.copy()
                        new_row[text_col] = augmented_text
                        new_row['augmentation_type'] = aug_type
                        augmented_rows.append(new_row)
        
        # Шум на уровне символов для всех примеров
        if self.apply_char_noise and self.char_noise_augmenter:
            print("Применение шума на уровне символов...")
            for idx, row in tqdm(df.iterrows(), desc='Добавление шума'):
                text = row[text_col]
                label = row[label_col]
                
                if not text or len(str(text).strip()) == 0:
                    continue
                
                # Определяем вероятность применения шума
                noise_prob = self.char_noise_prob
                if label == 1 or label == 1.0:  # Токсичный пример
                    noise_prob *= self.toxic_noise_multiplier
                    noise_prob = min(1.0, noise_prob)  # Не больше 1.0
                
                if random.random() < noise_prob:
                    # Определяем интенсивность шума
                    noise_intensity = self.char_noise_intensity
                    if label == 1 or label == 1.0:  # Больше шума для токсичных
                        noise_intensity *= self.toxic_noise_multiplier
                        noise_intensity = min(0.1, noise_intensity)  # Не больше 10%
                    
                    # Временно изменяем интенсивность
                    original_intensity = self.char_noise_augmenter.noise_intensity
                    self.char_noise_augmenter.noise_intensity = noise_intensity
                    
                    # Применяем случайный тип шума
                    augmented_text, aug_type = self.char_noise_augmenter.augment(str(text))
                    
                    # Восстанавливаем интенсивность
                    self.char_noise_augmenter.noise_intensity = original_intensity
                    
                    if augmented_text != text:  # Только если шум применен
                        new_row = row.copy()
                        new_row[text_col] = augmented_text
                        new_row['augmentation_type'] = aug_type
                        augmented_rows.append(new_row)
        
        # Добавляем аугментированные строки
        if augmented_rows:
            df_augmented = pd.DataFrame(augmented_rows)
            df_result = pd.concat([df_result, df_augmented], ignore_index=True)
        
        print(f"Аугментация завершена. Добавлено {len(augmented_rows)} примеров.")
        print(f"Итого: {len(df_result)} примеров (оригинальных: {len(df)}, аугментированных: {len(augmented_rows)})")
        
        # Статистика по типам аугментации
        if 'augmentation_type' in df_result.columns:
            aug_stats = df_result['augmentation_type'].value_counts()
            print("\nСтатистика по типам аугментации:")
            for aug_type, count in aug_stats.items():
                print(f"  {aug_type}: {count}")
        
        return df_result

