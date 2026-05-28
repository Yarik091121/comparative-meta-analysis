# data_processing.py
"""
Модуль загрузки данных, автоматической предобработки (ETL)
и удаления мультиколлинеарных признаков.
Содержит универсальную логику для работы с произвольными CSV и Excel файлами.
"""

import os
import pandas as pd
import numpy as np
from config import (
    CORRELATION_THRESHOLD, MISSING_THRESHOLD,
    CORRELATION_FALLBACK, EXCLUDE_FROM_FEATURES
)


def smart_read_file(filepath: str) -> pd.DataFrame:
    """
    Умное чтение файла с поддержкой CSV и Excel.
    Автоматически определяет разделитель, кодировку и наличие заголовков.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        # Перебираем возможные кодировки
        for encoding in ['utf-8-sig', 'utf-8', 'cp1251', 'latin1']:
            try:
                # sep=None с engine='python' позволяет pandas самому определить разделитель
                df = pd.read_csv(filepath, sep=None, engine='python', encoding=encoding)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        raise ValueError("Не удалось прочитать CSV файл. Проверьте формат и кодировку.")
    
    elif ext in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(filepath)
        except Exception as e:
            raise ValueError(f"Ошибка чтения Excel файла: {str(e)}")
    
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}. Используйте CSV или Excel.")


def get_data_overview(df: pd.DataFrame) -> dict:
    """
    Формирует обзор данных: размерность, типы, пропуски.
    """
    overview = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().mean() * 100).round(2).to_dict(),
        'head': df.head(5).to_dict('records'),
    }
    return overview


def identify_numeric_features(df: pd.DataFrame, target_col: str = None) -> list:
    """
    Возвращает список числовых столбцов, пригодных для анализа.
    Исключает ID, даты, целевую переменную и служебные столбцы.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(EXCLUDE_FROM_FEATURES)
    if target_col:
        exclude.add(target_col)
    
    return [c for c in numeric_cols if c not in exclude]


def preprocess_data(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Полный цикл предобработки:
    1. Заполнение пропусков (числовые → медиана, категориальные → мода)
    2. Удаление мультиколлинеарных признаков (|r| > threshold)
    
    Возвращает: (очищенный DataFrame, отчёт о преобразованиях)
    """
    report = []
    df_clean = df.copy()
    
    # === ШАГ 1: Обработка пропусков ===
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number, 'datetime']).columns
    
    # Числовые → медиана
    for col in num_cols:
        miss_count = df_clean[col].isnull().sum()
        if miss_count > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            report.append(f"📊 '{col}': заполнено {miss_count} пропусков медианой ({median_val:.2f})")
    
    # Категориальные → мода
    for col in cat_cols:
        miss_count = df_clean[col].isnull().sum()
        if miss_count > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
                report.append(f"🏷️ '{col}': заполнено {miss_count} пропусков модой ('{mode_val[0]}')")
    
    # === ШАГ 2: Удаление мультиколлинеарности ===
    features = identify_numeric_features(df_clean, target_col)
    dropped_cols = []
    
    if len(features) > 1:
        # Проверяем наличие target в данных
        if target_col not in df_clean.columns:
            report.append(f"⚠️ Целевая переменная '{target_col}' не найдена в данных")
            return df_clean, report
        
        corr_matrix = df_clean[features].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = set()
        threshold = CORRELATION_THRESHOLD
        
        for col in upper_tri.columns:
            correlated_pairs = upper_tri.index[upper_tri[col] > threshold].tolist()
            for row in correlated_pairs:
                miss_row = df_clean[row].isnull().mean()
                miss_col = df_clean[col].isnull().mean()
                
                # Если у обоих пропуски ≤ 10% — оставляем оба (согласно ТЗ)
                if miss_row <= MISSING_THRESHOLD and miss_col <= MISSING_THRESHOLD:
                    continue
                
                # Иначе удаляем столбец с большей долей пропусков
                if miss_row > miss_col:
                    to_drop.add(row)
                elif miss_col > miss_row:
                    to_drop.add(col)
                else:
                    # При равных пропусках — удаляем менее коррелированный с target
                    try:
                        corr_row = abs(df_clean[row].corr(df_clean[target_col]))
                        corr_col = abs(df_clean[col].corr(df_clean[target_col]))
                        if corr_row < corr_col:
                            to_drop.add(row)
                        else:
                            to_drop.add(col)
                    except Exception:
                        to_drop.add(col)
        
        if to_drop:
            df_clean = df_clean.drop(columns=list(to_drop))
            dropped_cols = list(to_drop)
            report.append(f"🗑️ Удалены мультиколлинеарные (|r|>{threshold}): {dropped_cols}")
        
        # Проверка: остались ли признаки
        remaining = identify_numeric_features(df_clean, target_col)
        if len(remaining) == 0:
            report.append(f"⚠️ Все признаки удалены! Попробуйте снизить порог до {CORRELATION_FALLBACK}")
    
    if not dropped_cols:
        report.append("✅ Мультиколлинеарных признаков не обнаружено")
    
    return df_clean, report


def compute_correlations(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Вычисляет корреляцию Пирсона всех числовых признаков с target.
    Возвращает отсортированный DataFrame.
    """
    features = identify_numeric_features(df, target_col)
    
    correlations = []
    for feat in features:
        try:
            mask = df[[feat, target_col]].notnull().all(axis=1)
            if mask.sum() > 2:
                r = df.loc[mask, feat].corr(df.loc[mask, target_col])
                correlations.append({
                    'Признак': feat,
                    'Корреляция (r)': round(r, 4),
                    '|r|': round(abs(r), 4)
                })
        except Exception:
            continue
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df = corr_df.sort_values('|r|', ascending=False).reset_index(drop=True)
    
    return corr_df


def encode_categorical(df: pd.DataFrame, columns_to_encode: list = None) -> pd.DataFrame:
    """
    One-Hot Encoding для категориальных столбцов.
    """
    if columns_to_encode is None:
        columns_to_encode = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
        # Исключаем служебные столбцы
        columns_to_encode = [c for c in columns_to_encode if c not in EXCLUDE_FROM_FEATURES]
    
    if columns_to_encode:
        df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    else:
        df_encoded = df.copy()
    
    return df_encoded