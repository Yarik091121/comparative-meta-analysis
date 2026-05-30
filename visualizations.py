# visualizations.py
"""
Модуль генерации графиков Plotly.
Адаптирован под собственный Fuzzy Engine.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_correlation_heatmap(corr_df: pd.DataFrame, target_col: str) -> go.Figure:
    """Горизонтальные бары корреляций."""
    if corr_df.empty:
        return go.Figure().add_annotation(text="Нет данных", showarrow=False)
    
    top = corr_df.head(15)
    fig = go.Figure(go.Bar(
        x=top['|r|'],
        y=top['Признак'],
        orientation='h',
        marker_color=px.colors.sequential.RdBu_r[:len(top)],
        text=top['Корреляция (r)'].round(3),
        textposition='outside',
    ))
    
    fig.update_layout(
        title=f'Топ-15 признаков по корреляции с "{target_col}"',
        xaxis_title='|Коэффициент корреляции|',
        yaxis=dict(autorange='reversed'),
        height=500,
        margin=dict(l=200),
    )
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name: str, r2: float) -> go.Figure:
    """Факт vs Прогноз."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        marker=dict(size=8, opacity=0.6, color='steelblue'),
        name='Данные'
    ))
    
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='y=x'
    ))
    
    fig.update_layout(
        title=f'{model_name}: Факт vs Прогноз (R²={r2:.4f})',
        xaxis_title='Факт',
        yaxis_title='Прогноз',
        height=450,
    )
    return fig


def plot_feature_importance(importances: dict, model_name: str) -> go.Figure:
    """Важность признаков."""
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig = go.Figure(go.Bar(
        x=[v for _, v in sorted_imp],
        y=[k for k, _ in sorted_imp],
        orientation='h',
        marker_color='teal',
    ))
    
    fig.update_layout(
        title=f'{model_name}: Важность признаков',
        xaxis_title='Важность',
        yaxis=dict(autorange='reversed'),
        height=450,
        margin=dict(l=200),
    )
    return fig


def plot_feature_dependence(df: pd.DataFrame, target_col: str, feature_col: str) -> go.Figure:
    """Scatter plot с непараметрической линией тренда."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[feature_col],
        y=df[target_col],
        mode='markers',
        marker=dict(color='steelblue', opacity=0.65),
        name='Наблюдения'
    ))

    mask = df[[feature_col, target_col]].notnull().all(axis=1)
    if mask.sum() >= 3:
        x = df.loc[mask, feature_col]
        y = df.loc[mask, target_col]
        sorted_idx = np.argsort(x)
        x_sorted = x.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]
        smooth = lowess(y_sorted, x_sorted, frac=0.3)
        fig.add_trace(go.Scatter(
            x=smooth[:, 0],
            y=smooth[:, 1],
            mode='lines',
            line=dict(color='red', width=2),
            name='Непараметрический тренд'
        ))

    fig.update_layout(
        title=f'Зависимость {target_col} от {feature_col} (Непараметрический тренд)',
        xaxis_title=feature_col,
        yaxis_title=target_col,
        height=450,
    )
    return fig


def plot_models_comparison(results: dict) -> go.Figure:
    """Сравнение R² моделей."""
    models = []
    r2_values = []
    colors = []
    color_map = {'OLS': '#1f77b4', 'Random Forest': '#ff7f0e', 'XGBoost': '#2ca02c'}
    
    for name in ['OLS', 'Random Forest', 'XGBoost']:
        if name in results and 'error' not in results[name]:
            models.append(name)
            r2_values.append(results[name]['r2'])
            colors.append(color_map.get(name, 'gray'))
    
    if not r2_values:
        return go.Figure().add_annotation(text="Нет данных", showarrow=False)
    
    fig = go.Figure(go.Bar(
        x=models, y=r2_values,
        marker_color=colors,
        text=[f'{v:.4f}' for v in r2_values],
        textposition='outside',
    ))
    
    fig.update_layout(
        title='Сравнение моделей: R²',
        yaxis_title='R²',
        yaxis_range=[0, max(r2_values) * 1.15] if r2_values else [0, 1],
        height=400,
    )
    return fig


def plot_metrics_table_figure(results: dict) -> go.Figure:
    """Таблица метрик."""
    models = ['OLS', 'Random Forest', 'XGBoost']
    metrics = ['R²', 'MAE', 'RMSE', 'MAPE (%)', 'Время (с)']
    
    values = []
    for m in models:
        if m in results and 'error' not in results[m]:
            r = results[m]
            values.append([
                f"{r['r2']:.4f}", f"{r['mae']:.2f}",
                f"{r['rmse']:.2f}", f"{r['mape']:.1f}%",
                f"{r['time']:.2f}"
            ])
        else:
            values.append(['—'] * 5)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Модель'] + metrics, fill_color='darkblue',
                    font=dict(color='white', size=12), align='center'),
        cells=dict(values=[models] + [[v[i] for v in values] for i in range(5)],
                   fill_color='lavender', font=dict(size=11), align='center')
    )])
    
    fig.update_layout(height=250, margin=dict(t=30, b=10))
    return fig


def plot_ols_diagnostics(ols_results: dict) -> go.Figure:
    """Диагностика OLS."""
    residuals = ols_results['residuals']
    fitted = ols_results['fitted']
    
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('Остатки vs Прогнозы', 'QQ-Plot', 'Гистограмма остатков'))
    
    fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers',
                             marker=dict(size=5, opacity=0.5)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                             marker=dict(size=5, color='coral')), row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=slope * np.array(osm) + intercept,
                             mode='lines', line=dict(color='red', dash='dash')), row=1, col=2)
    
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                               marker_color='lightblue'), row=1, col=3)
    
    fig.update_layout(title='Диагностика OLS', height=400, showlegend=False)
    return fig


def plot_fuzzy_membership(fuzzy_data: dict) -> go.Figure:
    """Графики функций принадлежности."""
    if 'error' in fuzzy_data:
        return go.Figure().add_annotation(text=fuzzy_data['error'], showarrow=False)
    
    feats = fuzzy_data['features']
    n_inputs = len(feats)
    fig = make_subplots(rows=1, cols=n_inputs + 1,
                        subplot_titles=feats + ['Выход (y_pred)'])
    
    colors = ['blue', 'green', 'red']
    term_labels = fuzzy_data['term_labels']
    
    # Входы
    for i, feat in enumerate(feats):
        var = fuzzy_data['antecedents'][feat]
        for j, term_name in enumerate(term_labels):
            mf = var.terms[term_name]
            fig.add_trace(go.Scatter(
                x=var.universe, y=mf,
                mode='lines',
                name=f'{feat}: {term_name}',
                line=dict(color=colors[j % 3], width=2),
                showlegend=(i == 0),
            ), row=1, col=i + 1)
    
    # Выход
    cons = fuzzy_data['consequent']
    for j, term_name in enumerate(term_labels):
        mf = cons.terms[term_name]
        fig.add_trace(go.Scatter(
            x=cons.universe, y=mf,
            mode='lines',
            name=f'Выход: {term_name}',
            line=dict(color=colors[j % 3], width=2, dash='dash'),
            showlegend=False,
        ), row=1, col=n_inputs + 1)
    
    fig.update_layout(title='Функции принадлежности', height=350)
    return fig


def plot_fuzzy_surface(fuzzy_data: dict, X_test: pd.DataFrame) -> go.Figure:
    """3D-поверхность отклика."""
    if 'error' in fuzzy_data or len(fuzzy_data['features']) < 2:
        return go.Figure().add_annotation(
            text="Нужно ≥ 2 признаков для 3D", showarrow=False)
    
    f1, f2 = fuzzy_data['features'][0], fuzzy_data['features'][1]
    fis = fuzzy_data['fis']
    
    x1_range = np.linspace(X_test[f1].min(), X_test[f1].max(), 25)
    x2_range = np.linspace(X_test[f2].min(), X_test[f2].max(), 25)
    
    # Фиксированные значения для остальных признаков
    fixed = {}
    for f in fuzzy_data['features'][2:]:
        if f in X_test.columns:
            fixed[f] = float(X_test[f].mean())
    
    _, _, Z = fis.compute_grid({f1: x1_range, f2: x2_range}, fixed)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    fig = go.Figure(data=[go.Surface(x=X1, y=X2, z=Z, colorscale='Viridis')])
    fig.update_layout(
        title=f'3D-поверхность: {f1} × {f2}',
        scene=dict(xaxis_title=f1, yaxis_title=f2, zaxis_title='Прогноз'),
        height=550,
    )
    return fig


def plot_fuzzy_contour(fuzzy_data: dict, X_test: pd.DataFrame) -> go.Figure:
    """Контурная карта."""
    if 'error' in fuzzy_data or len(fuzzy_data['features']) < 2:
        return go.Figure()
    
    f1, f2 = fuzzy_data['features'][0], fuzzy_data['features'][1]
    fis = fuzzy_data['fis']
    
    x1_range = np.linspace(X_test[f1].min(), X_test[f1].max(), 30)
    x2_range = np.linspace(X_test[f2].min(), X_test[f2].max(), 30)
    
    fixed = {}
    for f in fuzzy_data['features'][2:]:
        if f in X_test.columns:
            fixed[f] = float(X_test[f].mean())
    
    _, _, Z = fis.compute_grid({f1: x1_range, f2: x2_range}, fixed)
    
    fig = go.Figure(data=go.Contour(
        x=x1_range, y=x2_range, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Прогноз'),
    ))
    
    fig.update_layout(
        title=f'Контурная карта: {f1} × {f2}',
        xaxis_title=f1, yaxis_title=f2, height=450,
    )
    return fig


def plot_time_series_comparison(y_test, rf_pred, xgb_pred) -> go.Figure:
    """Сравнение прогнозов."""
    test_index = list(range(len(y_test)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=y_test, mode='lines+markers',
                             name='Факт', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=test_index, y=rf_pred, mode='lines',
                             name='RF', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=test_index, y=xgb_pred, mode='lines',
                             name='XGB', line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title='Сравнение прогнозов на тестовой выборке',
        xaxis_title='Номер наблюдения (хронологический порядок)', # Исправление п.1
        yaxis_title='Урожайность (кг/м²)', # Исправление п.1
        height=400
    )
    return fig


def plot_fuzzy_type_comparison(fuzzy1_data, fuzzy2_data, X_test):
    if 'error' in fuzzy1_data or 'error' in fuzzy2_data or len(fuzzy1_data['features']) < 2:
        return go.Figure().add_annotation(text="Недостаточно данных для сравнения", showarrow=False)
        
    f1, f2 = fuzzy1_data['features'][0], fuzzy1_data['features'][1]
    x1_range = np.linspace(X_test[f1].min(), X_test[f1].max(), 20)
    x2_range = np.linspace(X_test[f2].min(), X_test[f2].max(), 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    Z1 = np.zeros_like(X1)
    Z2 = np.zeros_like(X1)
    fixed = {f: float(X_test[f].mean()) for f in fuzzy1_data['features'][2:]}
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            inp = {**fixed, f1: X1[i,j], f2: X2[i,j]}
            Z1[i,j] = fuzzy1_data['fis'].compute(inp)
            Z2[i,j] = fuzzy2_data['fis'].compute(inp)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fuzzy Type-1', 'Fuzzy Type-2 (Interval)'),
        specs=[[{'type': 'scene'}, {'type': 'scene'}]]
    )
    fig.add_trace(go.Surface(x=X1, y=X2, z=Z1, colorscale='Blues', showscale=False), 1, 1)
    fig.add_trace(go.Surface(x=X1, y=X2, z=Z2, colorscale='Reds', showscale=False), 1, 2)
    fig.update_layout(title='Сравнение поверхностей отклика: Type-1 vs Type-2', height=500)
    return fig

def plot_copula_density(params, copula_name='Гауссова'):
    """
    Строит контурную карту плотности копулы.
    params: параметр(ы) копулы (rho для Гауссовой, theta для Clayton/Gumbel/Frank, (rho, nu) для Student-t)
    copula_name: название копулы
    """
    # Сетка от 0.01 до 0.99, чтобы избежать бесконечности на краях (ppf(0) = -inf)
    u = np.linspace(0.01, 0.99, 100)
    v = np.linspace(0.01, 0.99, 100)
    U, V = np.meshgrid(u, v)

    Z = np.zeros_like(U)

    if copula_name == 'Gaussian' or copula_name == 'Гауссова':
        rho = params if isinstance(params, float) else params[0]
        rho = np.clip(rho, -0.99, 0.99)
        # Преобразование в нормальные оценки (квантили)
        x = stats.norm.ppf(U)
        y = stats.norm.ppf(V)
        # Формула плотности Гауссовой копулы
        denominator = np.sqrt(1 - rho**2)
        exponent = -(rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
        Z = (1 / denominator) * np.exp(exponent)

    elif copula_name == 'Clayton':
        theta = params if isinstance(params, float) else params[0]
        theta = np.clip(theta, 0.01, 20)
        # Плотность копулы Клейтона: c(u,v) = (1+theta) * (u^(-theta) + v^(-theta) - 1)^(-(1/theta + 2)) * u^(-theta-1) * v^(-theta-1)
        Z = (1 + theta) * (U**(-theta) + V**(-theta) - 1)**(-(1/theta + 2)) * U**(-theta-1) * V**(-theta-1)

    elif copula_name == 'Gumbel':
        theta = params if isinstance(params, float) else params[0]
        theta = np.clip(theta, 1.01, 20)
        # Плотность копулы Гумбеля
        a = (-np.log(U))**theta + (-np.log(V))**theta
        # c(u,v) = C(u,v) * (a^(1/theta - 2)) * ((-ln u)^(theta-1)) * ((-ln v)^(theta-1)) * (1 + (theta-1)*a^(-1/theta)) / (u*v)
        # Упрощенная стабильная версия
        C = np.exp(-a**(1/theta))
        term1 = a**(1/theta - 2)
        term2 = ((-np.log(U)) * (-np.log(V)))**(theta - 1)
        term3 = 1 + (theta - 1) * a**(-1/theta)
        Z = C * term1 * term2 * term3 / (U * V)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    elif copula_name == 'Frank':
        theta = params if isinstance(params, float) else params[0]
        if abs(theta) < 1e-6:
            theta = 1e-6 # Избегаем деления на ноль
        # Плотность копулы Франка
        num = (1 - np.exp(-theta)) * np.exp(-theta * (U + V))
        den = (1 - np.exp(-theta*U) - np.exp(-theta*V) + np.exp(-theta*(U+V)))**2
        Z = num / den
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    elif copula_name == 'Student-t':
        if isinstance(params, tuple) and len(params) == 2:
            rho, nu = params
        else:
            rho, nu = params[0], 5.0 # fallback
        rho = np.clip(rho, -0.99, 0.99)
        nu = np.clip(nu, 2.1, 30)

        x = stats.t.ppf(U, nu)
        y = stats.t.ppf(V, nu)

        det_sigma = 1 - rho**2
        inv_sigma_quad = (x**2 - 2*rho*x*y + y**2) / det_sigma

        # Логарифм многомерной плотности t
        log_t2 = -0.5 * np.log(det_sigma) - ((nu + 2) / 2) * np.log(1 + inv_sigma_quad / nu)
        # Логарифм маргинальных плотностей
        log_t1_x = -((nu + 1) / 2) * np.log(1 + x**2 / nu)
        log_t1_y = -((nu + 1) / 2) * np.log(1 + y**2 / nu)

        log_c = log_t2 - log_t1_x - log_t1_y
        Z = np.exp(log_c)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # По умолчанию Гауссова
        rho = params if isinstance(params, float) else params[0]
        rho = np.clip(rho, -0.99, 0.99)
        x = stats.norm.ppf(U)
        y = stats.norm.ppf(V)
        denominator = np.sqrt(1 - rho**2)
        exponent = -(rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
        Z = (1 / denominator) * np.exp(exponent)

    fig = go.Figure(data=go.Contour(
        x=u, y=v, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Плотность'),
        contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
    ))
    
    fig.update_layout(
        title=f'Плотность копулы {copula_name}',
        xaxis_title='u (Ранг признака X)',
        yaxis_title='v (Ранг признака Y / Target)',
        height=500
    )
    return fig

def plot_veg_forecast_with_neighbors(model_name, prediction, X_test, y_test, input_vector, k=5):
    # Находим k ближайших соседей в тестовой выборке по евклидовому расстоянию
    dists = np.linalg.norm(X_test.values - np.array(input_vector).reshape(1, -1), axis=1)
    nearest_idx = np.argsort(dists)[:k]
    
    fig = go.Figure()
    # Соседи
    fig.add_trace(go.Scatter(
        x=list(range(k)), y=y_test.iloc[nearest_idx].values,
        mode='markers+text', marker=dict(size=12, color='blue', symbol='circle'),
        name=f'Ближайшие {k} точек (Тест)', text=[f"Факт: {v:.1f}" for v in y_test.iloc[nearest_idx].values], textposition="top center"
    ))
    # Прогноз
    fig.add_trace(go.Scatter(
        x=[k/2], y=[prediction], # По центру
        mode='markers+text', marker=dict(size=15, color='red', symbol='star'),
        name=f'Прогноз {model_name}', text=[f"Прогноз: {prediction:.1f}"], textposition="bottom center"
    ))
    
    fig.update_layout(
        title=f'Прогноз вегетации ({model_name}) и ближайшие исторические случаи',
        xaxis_title='Индекс соседства',
        yaxis_title='Длительность (дней)',
        xaxis=dict(showticklabels=False),
        height=350
    )
    return fig