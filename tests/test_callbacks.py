import pandas as pd
import numpy as np

from callbacks import apply_filters, _render_overview, _render_copula


def test_apply_filters_crop_greenhouse_variety():
    df = pd.DataFrame({
        'crop_type': ['wheat', 'wheat', 'corn'],
        'greenhouse_id': [1, 2, 1],
        'variety': ['v1', 'v2', 'v1'],
        'val': [10, 20, 30]
    })

    out = apply_filters(df, 'wheat', '1', ['v1'])
    # should select row with crop_type 'wheat', greenhouse_id '1' and variety 'v1'
    assert len(out) == 1
    assert out.iloc[0]['val'] == 10


def test_render_overview_reports_and_flags():
    results = {
        'report': ['step1', 'step2'],
        'veg_models': {'m': 1},
        'fuzzy': {'rules_count': 3, 'features': ['a', 'b']},
        'fuzzy2': {'rules_count': 0},
        'copula': {'Gaussian': {'param': 0.5, 'loglik': 1.0, 'aic': 2.0, 'bic': 3.0, 'tail_l': 0, 'tail_u': 0}},
        'train_size': 10,
        'test_size': 5,
    }

    comp = _render_overview(results)
    s = str(comp)
    assert 'Отчёт ETL' in s
    assert 'Train' in s or 'Train:' in s


def test_render_copula_none_and_with_data():
    # Case: copula missing
    res_none = {'copula': None}
    comp_none = _render_copula(res_none)
    assert 'Копула' in str(comp_none)

    # Case: copula present
    cop = {'Gaussian': {'param': 0.3, 'loglik': 0.0, 'aic': 1.0, 'bic': 1.5, 'tail_l': 0.0, 'tail_u': 0.0}}
    res = {'copula': cop}
    comp = _render_copula(res)
    s = str(comp)
    assert 'Анализ зависимостей' in s
