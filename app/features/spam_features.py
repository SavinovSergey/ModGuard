"""Ручные признаки и эвристики для классификации спама (сырой текст)."""
import re
from typing import List

import numpy as np

# Фиксированный порядок признаков для совместимости с пайплайном
SPAM_FEATURE_NAMES = [
    "caps_ratio",
    "url_count",
    "max_url_length_log1p",
    "has_short_domain",
    "has_suspicious_tld",
    "has_suspicious_params",
    "repeated_chars_ratio",
    "length_chars_log1p",
    "typo_score",
    "max_excl_run_log1p",
    "max_q_run_log1p",
    "upper_words_count",
    "upper_words_ratio",
    "space_ratio",
    "single_letter_token_ratio",
    "weird_char_ratio",
    "has_phone",
]

# URL: http(s), www, или явные короткие домены
URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|bit\.ly/\S+|tinyurl\.com/\S+|t\.co/\S+|goo\.gl/\S+",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
SHORT_DOMAINS = re.compile(
    r"bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|v\.gd|ow\.ly|adf\.ly",
    re.IGNORECASE,
)
SUSPICIOUS_TLD = re.compile(
    r"\.(xyz|top|work|click|link|info|biz|win|tk|ml|ga|cf|gq|cc|ws|buzz|rest|download|loan|money|stream)\b",
    re.IGNORECASE,
)
SUSPICIOUS_PARAMS = re.compile(r"\?(ref=|promo=|utm_|aff=)|&(ref=|utm_)", re.IGNORECASE)
# Слова с цифрой внутри (креди7т, беспл4тно)
DIGIT_IN_WORD = re.compile(
    r"[а-яА-Яa-zA-Z]+[0-9]+[а-яА-Яa-zA-Z]*|[а-яА-Яa-zA-Z]*[0-9]+[а-яА-Яa-zA-Z]+",
    re.IGNORECASE,
)
# Точки между буквами (б.е.с.п.л.а.т.н.о)
DOTS_INSIDE_WORD = re.compile(r"\b[a-zA-Zа-яА-Я][.]{1,2}[a-zA-Zа-яА-Я]+", re.IGNORECASE)

# Телефоноподобные последовательности (международные и российские форматы, 10–15 цифр)
PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)               # не внутри более длинной цифровой последовательности
    (?:\+?\d[\s\-\(\)]*)  # +7, 8, +1 и т.п. с разделителями
    (?:\d[\s\-\(\)]*){9,14}  # ещё 9–14 цифр с разделителями (в сумме 10–15 цифр)
    (?!\d)
    """,
    re.VERBOSE,
)

# Правило спама: слово целиком капсом (минимум 5 букв) + сразу после него не менее 3 восклицательных знаков
CAPS_WORD_DOUBLE_EXCL_PATTERN = re.compile(
    r"\b[A-ZА-ЯЁ]{5,}\s*!{2,}",
)
# Правило спама: слово из 7+ букв целиком капсом (недопустимо; аббревиатуры обычно короче)
CAPS_WORD_PATTERN = re.compile(r"\b[A-ZА-ЯЁ]{7,}\b")


def matches_caps_word_double_excl_rule(text: str) -> bool:
    """
    Правило спама: есть хотя бы одно слово целиком в верхнем регистре (≥2 букв),
    за которым следуют минимум 2 восклицательных знака (например БЕСПЛАТНО!!, СКИДКА!!!).
    """
    if not text or not text.strip():
        return False
    return bool(CAPS_WORD_DOUBLE_EXCL_PATTERN.search(text))


def matches_caps_word_rule(text: str) -> bool:
    """
    Правило спама: есть хотя бы одно слово из 7+ букв целиком в верхнем регистре
    (например БЕСПЛАТНО, ПОДАРОК). Порог 7 уменьшает срабатывание на короткие аббревиатуры.
    """
    if not text or not text.strip():
        return False
    return bool(CAPS_WORD_PATTERN.search(text))


def _find_urls(text: str) -> List[str]:
    return URL_PATTERN.findall(text)


def _caps_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _repeated_chars_ratio(text: str) -> float:
    """Доля символов, входящих в последовательности из 3+ одинаковых подряд."""
    if not text:
        return 0.0
    n = len(text)
    i = 0
    repeated_count = 0
    while i < n:
        j = i
        while j < n and text[j] == text[i]:
            j += 1
        if j - i >= 3:
            repeated_count += j - i
        i = j
    return repeated_count / n if n else 0.0


def _typo_score(text: str) -> float:
    """Эвристика опечаток: цифры внутри слов + точки между буквами."""
    if not text or not text.strip():
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    suspicious = 0
    for w in words:
        if DIGIT_IN_WORD.search(w) or DOTS_INSIDE_WORD.search(w):
            suspicious += 1
    return suspicious / len(words)


def _max_run_char(text: str, ch: str) -> int:
    """Максимальная длина последовательности символа ch подряд."""
    if not text:
        return 0
    n = len(text)
    i = 0
    max_run = 0
    while i < n:
        if text[i] != ch:
            i += 1
            continue
        j = i
        while j < n and text[j] == ch:
            j += 1
        run = j - i
        if run > max_run:
            max_run = run
        i = j
    return max_run


def _upper_words_stats(text: str) -> tuple[int, float]:
    """Количество и доля слов в верхнем регистре (с учётом кириллицы/латиницы)."""
    if not text or not text.strip():
        return 0, 0.0
    words = text.split()
    if not words:
        return 0, 0.0
    upper_count = 0
    letter_word_count = 0
    for w in words:
        core = w.strip(".,!?;:-()[]{}\"'…")
        if not core:
            continue
        has_letter = any(ch.isalpha() for ch in core)
        if not has_letter:
            continue
        letter_word_count += 1
        letters = [ch for ch in core if ch.isalpha()]
        if letters and all(ch.isupper() for ch in letters):
            upper_count += 1
    if letter_word_count == 0:
        return 0, 0.0
    return upper_count, upper_count / letter_word_count


def _space_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    spaces = text.count(" ")
    return spaces / total if total else 0.0


def _single_letter_token_ratio(text: str) -> float:
    """Доля одиночных буквенных токенов (Б.Е.С.П.Л.А.Т.Н.О и подобные)."""
    if not text or not text.strip():
        return 0.0
    raw_tokens = text.split()
    if not raw_tokens:
        return 0.0
    letter_tokens = 0
    single_letter_tokens = 0
    for tok in raw_tokens:
        core = tok.strip(".,!?;:-()[]{}\"'…")
        if not core:
            continue
        letters = [ch for ch in core if ch.isalpha()]
        if not letters:
            continue
        letter_tokens += 1
        if len(letters) == 1 and len(core) == 1:
            # полностью одиночная буква как отдельный токен
            single_letter_tokens += 1
    if letter_tokens == 0:
        return 0.0
    return single_letter_tokens / letter_tokens


def _weird_char_ratio(text: str) -> float:
    """Доля необычных символов (не буквы/цифры/пробел/базовая пунктуация)."""
    if not text:
        return 0.0
    allowed_punct = set(".,!?;:-()[]{}\"'…")
    total = len(text)
    if total == 0:
        return 0.0
    weird = 0
    for ch in text:
        if ch.isalpha() or ch.isdigit() or ch.isspace() or ch in allowed_punct:
            continue
        weird += 1
    return weird / total


def _phone_stats(text: str) -> tuple[int, float]:
    """Количество телефоноподобных последовательностей и бинарный флаг."""
    if not text or not text.strip():
        return 0, 0.0
    phones = PHONE_PATTERN.findall(text)
    count = len(phones)
    return count, 1.0 if count > 0 else 0.0


def extract_spam_features(text: str) -> dict:
    """Извлекает ручные признаки из сырого текста. Возвращает словарь с ключами из SPAM_FEATURE_NAMES."""
    if text is None:
        text = ""
    text = str(text).strip()
    urls = _find_urls(text)
    url_count = float(len(urls))
    max_url_len = max((len(u) for u in urls), default=0)
    max_url_length_log1p = np.log1p(max_url_len)
    url_str = " ".join(urls)
    has_short_domain = 1.0 if SHORT_DOMAINS.search(url_str) else 0.0
    has_suspicious_tld = 1.0 if SUSPICIOUS_TLD.search(url_str) else 0.0
    has_suspicious_params = 1.0 if SUSPICIOUS_PARAMS.search(text) else 0.0

    caps_ratio = _caps_ratio(text)
    repeated_chars_ratio = _repeated_chars_ratio(text)
    length_chars = len(text)
    length_chars_log1p = np.log1p(length_chars)
    typo_score = _typo_score(text)
    max_excl_run = _max_run_char(text, "!")
    max_q_run = _max_run_char(text, "?")
    max_excl_run_log1p = np.log1p(max_excl_run)
    max_q_run_log1p = np.log1p(max_q_run)
    upper_words_count, upper_words_ratio = _upper_words_stats(text)
    space_ratio = _space_ratio(text)
    single_letter_token_ratio = _single_letter_token_ratio(text)
    weird_char_ratio = _weird_char_ratio(text)
    _, has_phone = _phone_stats(text)

    return {
        "caps_ratio": float(caps_ratio),
        "url_count": url_count,
        "max_url_length_log1p": float(max_url_length_log1p),
        "has_short_domain": has_short_domain,
        "has_suspicious_tld": has_suspicious_tld,
        "has_suspicious_params": has_suspicious_params,
        "repeated_chars_ratio": float(repeated_chars_ratio),
        "length_chars_log1p": float(length_chars_log1p),
        "typo_score": float(typo_score),
        "max_excl_run_log1p": float(max_excl_run_log1p),
        "max_q_run_log1p": float(max_q_run_log1p),
        "upper_words_count": float(upper_words_count),
        "upper_words_ratio": float(upper_words_ratio),
        "space_ratio": float(space_ratio),
        "single_letter_token_ratio": float(single_letter_token_ratio),
        "weird_char_ratio": float(weird_char_ratio),
        "has_phone": float(has_phone),
    }


def extract_spam_features_batch(texts: List[str]) -> np.ndarray:
    """Батч: возвращает матрицу (n, len(SPAM_FEATURE_NAMES)) в порядке SPAM_FEATURE_NAMES."""
    rows = []
    for t in texts:
        d = extract_spam_features(t)
        rows.append([d[name] for name in SPAM_FEATURE_NAMES])
    return np.array(rows, dtype=np.float64)
