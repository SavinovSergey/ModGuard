"""Regex-модель для быстрого обнаружения спама (pre-filter перед TF-IDF)."""
import re
from typing import Any, Dict, List, Optional, Tuple


_SPAM_CATEGORIES: List[Tuple[str, re.Pattern]] = [
    # ── earnings: заработок / деньги (сильные маркеры) ─────────────────
    (
        "earnings",
        re.compile(
            r"\bбез\s*вложений\b"
            r"|(заработ\S*|доход\S*).{0,25}без\s*вложений"
            r"|без\s*вложений.{0,25}(заработ|доход)"
            r"|\bпассивн\S*\s+доход\b"
            r"|от\s+\d[\d\s]*\s*(руб\S*|₽|\$|€|тыс\S*|долл\S*)\s*(в\s+)?(день|час|сутк|недел|месяц)"
            r"|\bсхем\S*\s+(заработ|доход)"
            r"|\bвывод\S*\s+(на\s+)?(карт|кошел)"
            r"|(быстр|лёгк|легк)\S*\s+(заработ|деньг|доход)\b"
            r"|\bреферал\S*\s+(ссылк|код)|реферальн\S*\s+ссылк"
            r"|\bбонус\S*\s+за\s+(регистрац|рег)\b"
            r"|\bзаработ\S*\s+от\s+\d|\bдоход\S*\s+от\s+\d",
            re.IGNORECASE,
        ),
    ),
    # ── cta_links: призыв перейти / написать в лс ─────────────────────
    (
        "cta_links",
        re.compile(
            r"(перейд|переход|жми|нажми|кликай|тыкай)\S*\s+.{0,20}(ссылк|линк|link)\b"
            r"|подробност\S*\s+(в\s+)?(лс|личк|директ|л\.с)\b"
            r"|(пиши|напиши|написать)\s+(в\s+)?(лс|личк|директ)\b"
            r"|\bподпис\S*\s+на\s+канал\b|\bканал\S*\s+подпис",
            re.IGNORECASE,
        ),
    ),
    # ── casino: казино / ставки ───────────────────────────────────────
    (
        "casino",
        re.compile(
            r"(онлайн|online)\s*казино|казино\s*(онлайн|online)"
            r"|\bставк\S*\s+на\s+спорт\b"
            r"|\bбукмекер\S*\s+контор"
            r"|\bигров\S*\s+автомат"
            r"|удвой\S*\s+.{0,12}(депозит|баланс|ставк)",
            re.IGNORECASE,
        ),
    ),
    # ── crypto: крипто-заработок / инвестиции ─────────────────────────
    (
        "crypto",
        re.compile(
            r"заработ\S*.{0,15}(крипт|биткоин|bitcoin|эфир|ethereum)"
            r"|крипт\S*.{0,15}(заработ|инвестиц|доход|прибыл)"
            r"|инвестиц\S*.{0,15}(крипт|биткоин|bitcoin)"
            r"|(трейдинг|trading)\S*.{0,15}(доход|сигнал|обучен)",
            re.IGNORECASE,
        ),
    ),
    # ── urgency: срочность, набор людей ────────────────────────────────
    (
        "urgency",
        re.compile(
            r"осталось\s+\d+\s*(мест|штук|экземпляр)\b"
            r"|ограниченн\S*\s+предложени\s+только\b"
            r"|последн\S*\s+шанс\s+получи"
            r"|\bсрочно\s+нужен\S*\b|\bсрочно\s+нужны\S*\b"
            r"|\bсрочно\s+нужн[ао]\s+\w",
            re.IGNORECASE,
        ),
    ),
    # ── recruitment: ищу в команду, без опыта (типичный спам в комментах) ──
    (
        "recruitment",
        re.compile(
            r"\bищ(у|ем)\s+(тех\s+кто\b|(человека?|людей|сотрудник(a|ов)?|водител([ья]|ей)?)\s+(в\s+)?(команд[уеа]|коллектив)?)\b"
            r"|нужны\s+(люди|работники|сотрудники?)(\s+без\s+опыта)?"
            r"|\bнабор\s+(людей|сотрудников)?\s*(в\s+команду|на\s+работу)\b"
            r"|\bв\s+команд\S*\s+нужны\b|\bв\s+команд\S*\s+ищем\b"
            r"|\bтребу[ею]тся\s+\d*\s*(человека?|люд(и|ей)|сотрудник([иа]|ов))\b"
            r"|шабашк|халтурк?а|зарабатывай",  # шабашка или халтура
            re.IGNORECASE,
        ),
    ),
    # ── giveaway: раздачи / розыгрыши ─────────────────────────────────
    (
        "giveaway",
        re.compile(
            r"бесплатн\S*\s+(раздач|розыгрыш)\b"
            r"|розыгрыш\S*\s+(приз|подар|iphone|денег|телефон|гаджет)\b"
            r"|\bраздач\S*\s+(приз|бесплатн)",
            re.IGNORECASE,
        ),
    ),
    # ── suspicious_links: короткие ссылки ─────────────────────────────
    (
        "suspicious_links",
        re.compile(
            r"t\.me/[\w\-]+"
            r"|(bit\.ly|tinyurl\.com|clck\.ru|goo\.gl|is\.gd|v\.gd|ow\.ly)/\S+",
        ),
    ),
    # ── drugs: наркотики ──────────────────────────────────────────────
    (
        "drugs",
        re.compile(
            r"купи\S*\s+(гаш|мефедрон|кокаин|героин|спайс|марихуан)\b"
            r"|(закладк|клад)\S*.{0,15}(купи|заказ|доставк|район|адрес)",
            re.IGNORECASE,
        ),
    ),
    # ── adult: интим-услуги ───────────────────────────────────────────
    (
        "adult",
        re.compile(
            r"интим\S*\s+услуг\b"
            r"|эскорт\S*\s+(услуг|сопровожд)"
            r"|досуг\S*\s+для\s+взрослых\b",
            re.IGNORECASE,
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

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "spam_regex",
            "type": "regex",
            "is_loaded": self.is_loaded,
            "version": "1.0.0",
            "description": "Regex-based spam pre-filter",
            "categories_count": len(self.categories),
        }
