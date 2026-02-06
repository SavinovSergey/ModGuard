import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, f1_score)


from utils import prepare_text, find_obscene_words



DATA_DIR = './data/'

train = pd.read_parquet(DATA_DIR + 'train.parquet')

train['label'] = (train['label'] == 0).astype(np.int8)
train = train[train['text'] != '']
train['text_length'] = train['text'].str.len()
train['label'] = train['label'].astype(np.int8)
train['text_length'] = train['text_length'].astype(np.int32)


prepare_text(train)
find_obscene_words(train, 'lemmatized_text')