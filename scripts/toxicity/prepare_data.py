# Корень проекта (скрипт в scripts/toxicity/)
from pathlib import Path
import sys
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.preprocessing.text_processor import TextProcessor
from app.models.toxicity.regex_model import RegexModel
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def prepare_data_from_dataset(dataset: Dataset, text_col: str = 'text', label_col: str = 'label'):
    # инверсия меток для удобства
    # корректировка меток с учетом regex, считаем токсичным если regex нашел токсичность или метка в датасете 1
    dataset = dataset.map(lambda x: {text_col: x[text_col].lower()})
    text_processor = TextProcessor()
    dataset = dataset.map(lambda x: {'processed_text': text_processor.process(x[text_col])})
    regex_model = RegexModel()
    dataset = dataset.map(
        lambda x: {label_col: 1 if not x[label_col] or regex_model.predict(x['processed_text'])['is_toxic'] else 0})  
    return dataset


ds_name = 'Mnwa/russian-toxic'
print(f"Loading dataset {ds_name}...")
ds = load_dataset(ds_name)
print(f"Dataset {ds_name} loaded")

print(f"Preparing data for dataset {ds_name}...")
ds = prepare_data_from_dataset(ds)
print(f"Data for dataset {ds_name} prepared")

print("Split train/val/test and saving data...")
train_df, test_df = ds['train'].to_pandas(), ds['test'].to_pandas()
train_df = train_df[train_df['text'] != '']
train_df = train_df.drop_duplicates()
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['label'])

os.makedirs('./data/toxicity', exist_ok=True)
train_df.to_parquet("./data/toxicity/train.parquet")
val_df.to_parquet("./data/toxicity/val.parquet")
test_df.to_parquet("./data/toxicity/test.parquet")
print("Data saved")