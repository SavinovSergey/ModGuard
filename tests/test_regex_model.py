"""Тесты для Regex модели"""
import pytest
from app.models.regex_model import RegexModel


def test_regex_model_initialization():
    """Тест инициализации модели"""
    model = RegexModel()
    assert model.model_name == "regex"
    assert not model.is_loaded


def test_regex_model_load():
    """Тест загрузки модели"""
    model = RegexModel()
    model.load()
    assert model.is_loaded


def test_regex_model_predict_toxic():
    """Тест предсказания токсичного текста"""
    model = RegexModel()
    model.load()
    
    result = model.predict("это ебать какой-то текст")
    assert result['is_toxic'] is True
    assert result['toxicity_score'] == 1.0
    assert 'ебать' in result['toxicity_types']


def test_regex_model_predict_non_toxic():
    """Тест предсказания нетоксичного текста"""
    model = RegexModel()
    model.load()
    
    result = model.predict("это нормальный комментарий")
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0


def test_regex_model_predict_batch():
    """Тест batch предсказания"""
    model = RegexModel()
    model.load()
    
    texts = [
        "нормальный текст",
        "токсичный ебать текст"
    ]
    
    results = model.predict_batch(texts)
    assert len(results) == 2
    assert results[0]['is_toxic'] is False
    assert results[1]['is_toxic'] is True


def test_regex_model_info():
    """Тест получения информации о модели"""
    model = RegexModel()
    info = model.get_model_info()
    
    assert info['name'] == 'regex'
    assert info['type'] == 'regex'
    assert 'version' in info



