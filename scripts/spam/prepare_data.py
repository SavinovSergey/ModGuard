# Корень проекта (скрипт в scripts/spam/)
from pathlib import Path
import sys
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.preprocessing.text_processor import SpamTextProcessor
from app.models.spam.regex_model import SpamRegexModel
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def prepare_data_from_dataset(dataset: Dataset, text_col: str = 'text', label_col: str = 'label'):
    # инверсия меток для удобства
    # корректировка меток с учетом regex, считаем токсичным если regex нашел токсичность или метка в датасете 1
    dataset = dataset.map(lambda x: {text_col: x[text_col].lower()})
    text_processor = SpamTextProcessor()
    dataset = dataset.map(lambda x: {'processed_text': text_processor.process(x[text_col])})
    regex_model = SpamRegexModel()
    dataset = dataset.map(
        lambda x: {label_col: 1 if not x[label_col] or regex_model.predict(x['processed_text'])['is_toxic'] else 0})  
    return dataset


ds_name = 'benzlokzik/russian-spam-fork'
print(f"Loading dataset {ds_name}...")
ds = load_dataset(ds_name)
print(f"Dataset {ds_name} loaded")

# print(f"Preparing data for dataset {ds_name}...")
# ds = prepare_data_from_dataset(ds)
# print(f"Data for dataset {ds_name} prepared")

print("Split train/val/test and saving data...")
train_df, test_df = ds['train'].to_pandas(), ds['test'].to_pandas()
train_df = train_df[train_df['text'] != '']
train_df = train_df.drop_duplicates()
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

os.makedirs('./data/spam', exist_ok=True)
train_df.to_parquet("./data/spam/train.parquet")
val_df.to_parquet("./data/spam/val.parquet")
test_df.to_parquet("./data/spam/test.parquet")
print("Data saved")