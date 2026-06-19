"""Regex-модель для быстрого обнаружения спама (pre-filter перед TF-IDF)."""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.utils.regex_batch import PatternLabel, batch_regex_classify

_FLAGS = re.IGNORECASE

# Pareto-набор: recruitment + earnings + cta_links
_SPAM_CATEGORIES: List[Tuple[str, re.Pattern]] = [
    (
        "earnings",
        re.compile(
            r"\bбез\s*вложений\b"
            r"|(?:заработ\w*|доход\w*)[^\n]{0,25}без\s*вложений"
            r"|без\s*вложений[^\n]{0,25}(?:заработ|доход)"
            r"|\bпассивн\w*\s+доход\b"
            r"|\bсхем\w*\s+(?:заработ|доход)"
            r"|\b(?:реферал\w*\s+(?:ссылк|код)|реферальн\w*\s+ссылк)"
            r"|\bбонус\w*\s+за\s+(?:регистрац|рег)\b"
            r"|\b(?:заработ|доход)\w*\s+от\s+\d"
            r"|\bвывод\w*\s+(?:на\s+)?(?:карт|кошел)",
            _FLAGS,
        ),
    ),
    (
        "cta_links",
        re.compile(
            r"(?:перейд|переход|(на)?жми|кликай|тыкай)\w*[^\n]{0,20}(?:ссылк|линк|link)\b"
            r"|\b(?:(на)?пиши)\s+(?:в\s+)?(?:лс|личк|директ)\b"
            r"|\bподпис\w*\s+на\s+канал\b"
            r"|\bканал\w*\s+подпис",
            _FLAGS,
        ),
    ),
    (
        "recruitment",
        re.compile(
            r"шабашк|"
            r"халтурк?а|"
            r"зарабатывай",
            _FLAGS,
        ),
    ),
]


class SpamRegexModel:
    """Быстрый regex-фильтр спама (pre-filter перед ML-моделью)."""

    def __init__(self) -> None:
        self.categories = _SPAM_CATEGORIES
        self.is_loaded = False

    def load(self, model_path: Optional[str] = None) -> None:
        self.is_loaded = True

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"is_spam": False, "spam_score": 0.0}

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return self._empty()

        matched: Dict[str, float] = {}
        for name, pattern in self.categories:
            if pattern.search(text):
                matched[name] = 1.0

        if not matched:
            return self._empty()

        return {
            "is_spam": True,
            "spam_score": 1.0,
            "spam_categories": matched,
        }

    def _labeled_patterns(self) -> Sequence[PatternLabel]:
        return list(self.categories)

    @staticmethod
    def _hit_result(categories: Dict[str, float]) -> Dict[str, Any]:
        return {
            "is_spam": True,
            "spam_score": 1.0,
            "spam_categories": categories,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return batch_regex_classify(
            texts,
            self._labeled_patterns(),
            empty_result=self._empty,
            hit_result=self._hit_result,
            pool_tag="spam",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "spam_regex",
            "type": "regex",
            "is_loaded": self.is_loaded,
            "version": "1.0.0",
            "description": "Regex-based spam pre-filter (recruitment+earnings+cta_links)",
            "categories_count": len(self.categories),
        }
