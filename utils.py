import re
from typing import List
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, f1_score)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from emoji import demojize
from pymorphy3 import MorphAnalyzer
from tqdm import tqdm

tqdm.pandas()


morph = MorphAnalyzer()

@lru_cache(maxsize=100000) 
def lemma(w):
    return morph.parse(w)[0].normal_form

def emoji_to_word(text):
    return demojize(text, language='ru')

def lemmatize(text):
    words = text.split() # разбиваем текст на слова
    res = [lemma(w) for w in words]
    return ' '.join(res)

stop_words = set(stopwords.words('russian'))
def drop_stop_words(data: List[str]):
    return ' '.join([w for w in data.split() if not w in stop_words])

def prepare_text(df):
    df['text'] = df['text'].str.lower()
    # удаляем ссылки на сайты и почты и аккаунты
    df['clean_text'] = df['text'].str.replace(r"http\S+|[a-z\d\._-]+@[a-z\d\._-]+\.[a-z\d\._-]+|@[a-z]+", " ", regex=True)
    # HTML тэги
    df['clean_text'] = df['clean_text'].str.replace(r"<.*?>", " ", regex=True)
    # Удаление отметок вида [id647188941|зара],
    df['clean_text'] = df['clean_text'].str.replace(r"\[id\d+|.+\], ", "", regex=True)
    # Удаление сочетаний вида &#33;
    df['clean_text'] = df['clean_text'].str.replace(r'&#\d+;|&.+;', ' ', regex=True)
    # удалим номера заказов
    df['clean_text'] = df['clean_text'].str.replace(r"[a-z\d]{7,8}-[a-z\d]{4}-[a-z\d]{4}-[a-z\d]{4}-[a-z\d]{12}|[\s^][№\d]+(\s|$)", " ", regex=True)
    # print('Обработка эмодзи.')
    # df['clean_text'] = df['clean_text'].progress_apply(emoji_to_word)
    df['clean_text'] = df['clean_text'].str.replace(r'\s{2,}', ' ', regex=True)
    print('Лемматизация.')
    df['lemmatized_text'] = df['clean_text'].progress_apply(lemmatize)
    df['lemmatized_text'] = df['lemmatized_text'].str.replace('ё', 'е')
    print('Удаление стоп-слов.')
    df['lemmatized_text'] = df['lemmatized_text'].progress_apply(drop_stop_words)


def find_obscene_words(df, text_col='text'):
    reg1 = re.compile(r"\bу?еб[алуи]?\b|збc|[зн]аеб[^р]\S*|\b(ебн?у|(по|[нз]а)?(еб|ип)[иаеу]?ть)\S*|[зд][ъь]еб|ебла|еб[еы]й|\bеба[^й]|еб[ау](л|ть)|\bебет|[еи][бп]ану|выеб")
    reg2 = re.compile(r"\b((по|ни|на|а|о)?ху[ейяию]|аху)\S*|\bхеров|херн|\b(хули|ху?[\sй]?н[яюе]|х\s?у?ета?|хер)\b|титьк|сиськ")
    reg3 = re.compile(r"\bбл[яеэ]+([тд]ь?)?\b|бляд|жоп|залуп\S*|трах[ан]|г[ао]ндон|д[еи]бил|чь?мо|идиот|ублюд|шлюх|урод|д[оа]лб[aо]")
    reg4 = re.compile(r"п[еи]?зде?ц?|\bпиз\b|пид[оа]?р|\bтрах|баба")
    reg5 = re.compile(r"г[ао]вн|\b(дерьмо|г.мно|гуано)\b|[на|по]?ср[ае](ть|[нл])|выс(ирать|ер)")
    reg6 = re.compile(r"\bтвар[иь]\b|мудак|сволочь|дрянь|(рас|от)стрел|дроч|мраз|суч?ка|сосать|нассать|минет|шмара|гнида|проститутка|придурок|даун|пиндос|безмозгл|козел")
    obscene_cols = ['ебать', 'хуй', 'бля', 'пиздец', 'говно', 'прочее']
    for reg, obs in zip([reg1, reg2, reg3, reg4, reg5, reg6], obscene_cols):
        df[f'is_{obs}'] = df[text_col].str.contains(reg, regex=True)
    # Для последующего сравнения, что лучше работает. 1 общий столбец или 4 по каждому мату.
    df['is_obscene'] = df[[f'is_{obs}' for obs in obscene_cols]].any(axis=1).astype(np.int8)