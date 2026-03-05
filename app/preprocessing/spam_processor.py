"""Лёгкая предобработка текста для детекции спама: без удаления цифр и email (сигналы спама)."""
import re
from typing import List, Tuple


# Базовый список русских стоп-слов для TF-IDF (эвристики считаются по сырому тексту)
RUSSIAN_STOPWORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "её",
    "их",
    "или",
    "еще",
    "из",
    "ли",
    "же",
    "ну",
    "от",
    "до",
    "для",
    "это",
    "если",
    "при",
    "там",
    "тут",
}


def _is_caps_token(token: str) -> bool:
    """Слово считаем капсом, если хотя бы один непервый символ — заглавная (пРиВеТ, БЕСПЛАТНО)."""
    core = token.strip(".,!?;:-()[]{}\"'…")
    if not core:
        return False
    letters = [c for c in core if c.isalpha()]
    if len(letters) < 2:
        return False
    return any(letters[i].isupper() for i in range(1, len(letters)))


def split_caps_rest(processed_text: str) -> Tuple[str, str]:
    """
    Делит предобработанный текст на капс-часть и остаток.
    Капс: токены, где есть хотя бы одна заглавная не на первой позиции (без приведения к нижнему).
    Остаток: остальные токены, приводятся к нижнему регистру.
    """
    if not processed_text or not processed_text.strip():
        return "", ""
    tokens = processed_text.split()
    caps_tokens: List[str] = []
    rest_tokens: List[str] = []
    for tok in tokens:
        if _is_caps_token(tok):
            caps_tokens.append(tok)
        else:
            rest_tokens.append(tok.lower())
    return " ".join(caps_tokens), " ".join(rest_tokens)


def split_caps_rest_batch(processed_texts: List[str]) -> Tuple[List[str], List[str]]:
    """Батч: (caps_parts, rest_parts) — списки строк для двух векторизаторов."""
    caps_parts = []
    rest_parts = []
    for t in processed_texts:
        c, r = split_caps_rest(t)
        caps_parts.append(c)
        rest_parts.append(r)
    return caps_parts, rest_parts


class SpamTextProcessor:
    """
    Предобработка для спама: нормализация без удаления ссылок/email/цифр.
    Для токсичности используется TextProcessor с более жёсткой очисткой.
    """

    def process(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            return ""
        s = text.strip()
        if not s:
            return ""

        # s = s.lower()     # капс - сильный сигнал для спама
        # Удаляем только HTML-теги
        s = re.sub(r"<[^>]+>", " ", s)
        # Удаляем отметки вида [id123|name]
        s = re.sub(r"\[id\d+\|[^\]]*\],?\s*", "", s)
        s = re.sub(r"&#\d+;|&[a-z]+;", " ", s)
        # Убираем лишние пробелы (цифры, email, URL оставляем)
        s = re.sub(r"\s{2,}", " ", s)

        # Удаляем русские стоп-слова только для TF-IDF (эвристики считаются по сырому тексту)
        tokens = s.split()
        if tokens:
            filtered = []
            for tok in tokens:
                tl = tok.lower()
                if tl in RUSSIAN_STOPWORDS:
                    continue
                filtered.append(tok)
            s = " ".join(filtered)

        return s.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(t) for t in texts]
