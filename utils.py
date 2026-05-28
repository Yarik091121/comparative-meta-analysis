# utils.py
"""
Вспомогательные утилиты: хеширование, метрики, логирование.
"""

import hashlib
import time
import numpy as np
from datetime import datetime


def get_file_hash(filepath: str) -> str:
    """
    Генерирует уникальный ключ кэша на основе пути к файлу и текущей даты.
    Ключ меняется каждый день, что обеспечивает ежедневную инвалидацию.
    """
    m = hashlib.md5()
    m.update(filepath.encode('utf-8'))
    # Привязка к дате: mm_dd_yyyy
    m.update(time.strftime("%m_%d_%Y").encode('utf-8'))
    return m.hexdigest()


def get_analysis_cache_key(file_hash: str, target: str, features: list) -> str:
    """
    Ключ для кэширования результатов анализа.
    Зависит от файла + target + набора признаков.
    """
    features_str = "_".join(sorted(features))
    raw = f"{file_hash}_{target}_{features_str}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def calculate_mape(y_true, y_pred) -> float:
    """MAPE с защитой от деления на ноль."""
    eps = 1e-8
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def format_timestamp() -> str:
    """Текущее время в формате для логов."""
    return datetime.now().strftime("%H:%M:%S")


def safe_float(value, default=0.0) -> float:
    """Безопасное преобразование к float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_column_safe(df, candidates: list, default=None):
    """
    Возвращает первое имя столбца из candidates, которое есть в DataFrame.
    Полезно для работы с русскими названиями.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return default