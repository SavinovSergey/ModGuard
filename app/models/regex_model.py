"""Regex модель для классификации токсичности на основе регулярных выражений"""
import re
from typing import List, Dict, Any

from app.models.base import BaseToxicityModel
from app.preprocessing.text_processor import TextProcessor


class RegexModel(BaseToxicityModel):
    """Модель классификации токсичности на основе регулярных выражений"""
    
    def __init__(self):
        super().__init__("regex")
        self.patterns = self._compile_patterns()
        self.toxicity_types = ['ебать', 'хуй', 'бля', 'пиздец', 'говно', 'прочее']
        self.text_processor = TextProcessor()
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Компилирует регулярные выражения для поиска нецензурных слов"""
        patterns = [
            # ебать
            re.compile(
                r"\bу?еб[алуи]?\b|збc|[зн]аеб[^р]\S*|\b(ебн?у|(по|[нз]а)?(еб|ип)[иаеу]?ть)\S*|"
                r"[зд][ъь]еб|ебла|еб[еы]й|\bеба[^й]|еб[ау](л|ть)|\bебет|[еи][бп]ану|выеб"
            ),
            # хуй
            re.compile(
                r"\b((по|ни|на|а|о)?ху[ейяию]|аху)\S*|\bхеров|херн|\b(хули|ху?[\sй]?н[яюе]|"
                r"х\s?у?ета?|хер)\b|титьк|сиськ"
            ),
            # бля
            re.compile(
                r"\bбл[яеэ]+([тд]ь?)?\b|бляд|жоп|залуп\S*|трах[ан]|г[ао]ндон|д[еи]бил|"
                r"чь?мо|идиот|ублюд|шлюх|урод|д[оа]лб[aо]"
            ),
            # пиздец
            re.compile(r"п[еи]?зде?ц?|\bпиз\b|пид[оа]?р|\bтрах|баба"),
            # говно
            re.compile(
                r"г[ао]вн|\b(дерьмо|г.мно|гуано)\b|[на|по]?ср[ае](ть|[нл])|выс(ирать|ер)"
            ),
            # прочее
            re.compile(
                r"\bтвар[иь]\b|мудак|сволочь|дрянь|(рас|от)стрел|дроч|мраз|суч?ка|"
                r"сосать|нассать|минет|шмара|гнида|проститутка|придурок|даун|пиндос|"
                r"безмозгл|козел"
            ),
        ]
        return patterns
    
    def load(self, model_path: str = None) -> None:
        """Загружает модель (для regex модели не требуется загрузка из файла)"""
        self.is_loaded = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает токсичность для одного текста
        
        Args:
            text: Предобработанный текст
        
        Returns:
            Словарь с результатами классификации
        """
        if not text:
            return {
                'is_toxic': False,
                'toxicity_score': 0.0,
                'toxicity_types': {}
            }
        
        # Предобработка текста
        text = self.text_processor.process(text)

        # Проверяем каждый паттерн
        toxicity_types = {}
        is_toxic = False
        
        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                type_name = self.toxicity_types[i]
                toxicity_types[type_name] = 1.0
                is_toxic = True
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': 1.0 if is_toxic else 0.0,
            'toxicity_types': toxicity_types
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказывает токсичность для батча текстов
        
        Args:
            texts: Список предобработанных текстов
        
        Returns:
            Список словарей с результатами классификации
        """
        processed_texts = self.text_processor.process_batch(texts)
        return [self.predict(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        return {
            'name': self.model_name,
            'type': 'regex',
            'is_loaded': self.is_loaded,
            'version': '1.0.0',
            'description': 'Regex-based toxicity classification model',
            'patterns_count': len(self.patterns)
        }




