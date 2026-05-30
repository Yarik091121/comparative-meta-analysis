import numpy as np
import pandas as pd
import plotly.graph_objects as go

from visualizations import (
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_models_comparison,
    plot_metrics_table_figure,
    plot_ols_diagnostics,
    plot_fuzzy_membership,
    plot_fuzzy_surface,
    plot_fuzzy_contour,
    plot_feature_dependence,
    plot_fuzzy_type_comparison,
    plot_copula_density,
    plot_veg_forecast_with_neighbors,
)

from fuzzy_engine import build_fuzzy_from_data, build_fuzzy_type2_from_data


def test_plot_correlation_heatmap_empty():
    df = pd.DataFrame()
    fig = plot_correlation_heatmap(df, 'target')
    # should contain an annotation saying "Нет данных"
    ann = getattr(fig.layout, 'annotations', None)
    assert ann is not None and len(ann) > 0
    assert 'Нет данных' in ann[0].text


def test_plot_actual_vs_predicted_and_feature_importance():
    y_test = [1.0, 2.0, 3.0]
    y_pred = [0.9, 2.1, 2.9]
    fig = plot_actual_vs_predicted(y_test, y_pred, 'ModelX', 0.95)
    assert len(fig.data) >= 2
    assert 'ModelX' in fig.layout.title.text

    imp = {'a': 0.2, 'b': 0.8}
    fig2 = plot_feature_importance(imp, 'RF')
    assert fig2.data[0].type == 'bar'
    # y labels should be sorted by importance (b then a)
    assert list(fig2.data[0].y) == ['b', 'a']


def test_plot_feature_dependence():
    np.random.seed(1)
    n = 30
    df = pd.DataFrame({
        'feature': np.linspace(0, 10, n),
        'target': np.linspace(1, 20, n) + np.random.randn(n) * 0.5
    })
    fig = plot_feature_dependence(df, 'target', 'feature')
    assert fig.data[0].type == 'scatter'
    assert any(trace.name == 'Непараметрический тренд' for trace in fig.data)
    assert 'Зависимость target от feature' in fig.layout.title.text


def test_plot_models_comparison_and_metrics_table():
    empty = {}
    fig_empty = plot_models_comparison(empty)
    assert getattr(fig_empty.layout, 'annotations', None) is not None

    results = {
        'OLS': {'r2': 0.9},
        'Random Forest': {'r2': 0.8},
    }
    fig = plot_models_comparison(results)
    assert fig.data[0].type == 'bar'
    assert 'OLS' in list(fig.data[0].x)

    metrics = {
        'OLS': {'r2': 0.9, 'mae': 0.1, 'rmse': 0.2, 'mape': 5.0, 'time': 0.12},
    }
    fig_table = plot_metrics_table_figure(metrics)
    assert fig_table.data[0].type == 'table'
    assert 'Модель' in fig_table.data[0].header.values[0]


def test_plot_ols_diagnostics_basic():
    np.random.seed(0)
    res = {'residuals': np.random.randn(100), 'fitted': np.random.randn(100)}
    fig = plot_ols_diagnostics(res)
    assert 'Диагностика OLS' in fig.layout.title.text


def test_plot_fuzzy_visualizations_and_type_comparison():
    # build fuzzy systems with 2 features
    np.random.seed(3)
    n = 80
    X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n), 'f3': np.random.randn(n)})
    y = X['f1'] * 1.5 + np.random.randn(n) * 0.1

    f1 = build_fuzzy_from_data(X, y, ['f1', 'f2', 'f3'], max_features=2)
    f2 = build_fuzzy_type2_from_data(X, y, ['f1', 'f2', 'f3'], max_features=2)

    fig_mem = plot_fuzzy_membership(f1)
    assert isinstance(fig_mem, go.Figure)

    X_test = X.copy()
    fig_surf = plot_fuzzy_surface(f1, X_test)
    assert isinstance(fig_surf, go.Figure)
    # first trace should be a surface
    assert fig_surf.data[0].type == 'surface'

    fig_cont = plot_fuzzy_contour(f1, X_test)
    assert isinstance(fig_cont, go.Figure)
    assert fig_cont.data[0].type in ('contour', 'heatmap')

    fig_cmp = plot_fuzzy_type_comparison(f1, f2, X_test)
    assert isinstance(fig_cmp, go.Figure)


def test_plot_copula_density_and_veg_forecast():
    fig = plot_copula_density(0.5, copula_name='Gaussian')
    assert fig.data[0].type == 'contour'
    z = np.array(fig.data[0].z)
    assert z.shape[0] == 100 and z.shape[1] == 100

    # veg forecast plot
    X_test = pd.DataFrame({'a': np.linspace(0, 1, 10), 'b': np.linspace(1, 2, 10)})
    y_test = pd.Series(np.linspace(5, 14, 10))
    input_vector = [0.5, 1.5]
    figv = plot_veg_forecast_with_neighbors('M', 7.5, X_test[['a','b']], y_test, input_vector, k=3)
    assert len(figv.data) == 2
    assert 'M' in figv.layout.title.text
