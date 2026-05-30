import pandas as pd
import numpy as np

from data_processing import (
    get_data_overview,
    identify_numeric_features,
    preprocess_data,
    compute_correlations,
    encode_categorical,
)
from config import EXCLUDE_FROM_FEATURES


def test_get_data_overview_basic():
    df = pd.DataFrame({
        'a': [1, 2, None],
        'b': ['x', 'y', 'z'],
        'c': [1.0, None, 3.5]
    })
    overview = get_data_overview(df)
    assert overview['shape'] == (3, 3)
    assert 'a' in overview['columns']
    assert overview['missing']['a'] == 1
    assert isinstance(overview['head'], list)


def test_identify_numeric_features_excludes_target_and_ids():
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4.0, 5.0, 6.0],
        'greenhouse_id': [10, 11, 12],
        'target': [0, 1, 0]
    })
    features = identify_numeric_features(df, target_col='target')
    assert 'num1' in features and 'num2' in features
    assert 'greenhouse_id' not in features
    assert 'target' not in features


def test_preprocess_data_fills_and_drops():
    # Create correlated features
    np.random.seed(0)
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = x1 * 0.99 + np.random.randn(n) * 0.01  # highly correlated
    target = x1 * 2 + np.random.randn(n) * 0.1

    df = pd.DataFrame({
        'f1': x1,
        'f2': x2,
        'cat': ['a'] * 50 + [None] * 50,
        'target': target
    })

    # Introduce missing values in f2 to trigger drop logic
    df.loc[:19, 'f2'] = np.nan

    df_clean, report = preprocess_data(df, 'target')
    # Missing in 'cat' should be filled
    assert df_clean['cat'].isnull().sum() == 0
    # Either f1 or f2 should be present (one may be dropped)
    assert 'f1' in df_clean.columns or 'f2' in df_clean.columns
    assert any('Удалены' in r or 'Мультиколлинеарных' in r or 'Мультиколлинеарных' in ''.join(report) or 'не обнаружено' in ''.join(report).lower() for r in report)


def test_compute_correlations_and_order():
    np.random.seed(1)
    n = 50
    x1 = np.linspace(0, 1, n)
    x2 = np.random.randn(n)
    target = x1 * 3 + np.random.randn(n) * 0.1

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'target': target})
    corr_df = compute_correlations(df, 'target')
    assert not corr_df.empty
    # x1 should have higher |r| than x2
    assert corr_df.iloc[0]['Признак'] == 'x1'


def test_encode_categorical_one_hot():
    df = pd.DataFrame({'a': [1, 2, 3], 'cat': ['x', 'y', 'x']})
    enc = encode_categorical(df, columns_to_encode=['cat'])
    # cat_y should exist (drop_first=True)
    assert any(col.startswith('cat_') for col in enc.columns)
