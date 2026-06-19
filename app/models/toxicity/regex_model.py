"""Regex модель для классификации токсичности на основе регулярных выражений"""
import re
from typing import List, Dict, Any, Sequence

from app.preprocessing.text_processor import TextProcessor
from app.models.base import ClassicalTextModelBase
from app.utils.regex_batch import PatternLabel, batch_regex_classify

_FLAGS = re.IGNORECASE


class RegexModel(ClassicalTextModelBase):
    """Модель классификации токсичности на основе регулярных выражений"""

    def __init__(self):
        super().__init__(model_name="regex", model_type="regex")
        self.toxicity_types = ["хуй", "бля", "пиздец", "говно"]
        self.patterns = self._compile_patterns()
        self.text_processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)

    def _compile_patterns(self) -> List[re.Pattern]:
        """Pareto-набор: хуй, бля, пиздец, говно (без ебать/прочее)."""
        return [
            # хуй (исходный паттерн)
            re.compile(
                r"\b(?:(?:по|ни|на|а|о)?ху[ейяию]|аху)\S*|"
                r"\bхеров|херн|"
                r"\b(?:хули|ху?[\sй]?н[яюе]|х\s?у?ета?|хер)\b|"
                r"титьк|сиськ",
                _FLAGS,
            ),
            # бля (исходный, без «трах» — он в пиздец)
            re.compile(
                r"\bбл[яеэ]+(?:[тд]ь?)?\b|"
                r"бляд|"
                r"жоп|"
                r"залуп\S*|"
                r"г[ао]ндон|"
                r"д[еи]бил|"
                r"ч[ьъ]?мо|"
                r"идиот|"
                r"ублюд|"
                r"шлюх|"
                r"урод|"
                r"д[оа]лб[ао]",
                _FLAGS,
            ),
            # пиздец (исходный)
            re.compile(
                r"п[еи]?зде?ц?|"
                r"\bпиз\b|"
                r"пид[оа]?р|"
                r"\bтрах",
                _FLAGS,
            ),
            # говно (исходный)
            re.compile(
                r"г[ао]вн|"
                r"\bдерьмо\b",
                _FLAGS,
            ),
        ]

    def _labeled_patterns(self) -> Sequence[PatternLabel]:
        return [(self.toxicity_types[i], pattern) for i, pattern in enumerate(self.patterns)]

    @staticmethod
    def _hit_result(categories: Dict[str, float]) -> Dict[str, Any]:
        return {
            "is_toxic": True,
            "toxicity_score": 1.0,
            "toxicity_types": categories,
        }

    def load(self, model_path: str = None) -> None:
        """Загружает модель (для regex модели не требуется загрузка из файла)"""
        self.is_loaded = True

    def predict(self, text: str) -> Dict[str, Any]:
        if not text:
            return self.empty_result()

        toxicity_types = {}
        is_toxic = False

        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                type_name = self.toxicity_types[i]
                toxicity_types[type_name] = 1.0
                is_toxic = True

        return {
            "is_toxic": is_toxic,
            "toxicity_score": 1.0 if is_toxic else 0.0,
            "toxicity_types": toxicity_types,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return batch_regex_classify(
            texts,
            self._labeled_patterns(),
            empty_result=self.empty_result,
            hit_result=self._hit_result,
            pool_tag="tox",
        )

    def get_model_info(self) -> Dict[str, Any]:
        info = self._base_info("Regex-based toxicity classification model")
        info["patterns_count"] = len(self.patterns)
        return info
