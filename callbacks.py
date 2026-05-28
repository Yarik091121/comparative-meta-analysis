# callbacks.py
"""
Все callback-функции приложения.
Исправлено для Dash 2.17+ (background_callback_manager).
"""

import os
import base64
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc

from config import TEMP_UPLOAD_DIR, TARGET_DEFAULT, EXCLUDE_FROM_FEATURES
from utils import get_file_hash, get_analysis_cache_key, format_timestamp
from cache_manager import cache_manager, GLOBAL_MEMORY
from data_processing import (
    smart_read_file, get_data_overview, preprocess_data,
    compute_correlations, identify_numeric_features, encode_categorical
)
from models import (
    train_ols, train_random_forest, train_xgboost,
    build_fuzzy_system, train_vegetation_model
)
from fuzzy_engine import build_fuzzy_type2_from_data
from copula_engine import CopulaAnalyzer, GaussianCopulaRegressor
from visualizations import (
    plot_correlation_heatmap, plot_actual_vs_predicted,
    plot_feature_importance, plot_models_comparison,
    plot_ols_diagnostics, plot_fuzzy_membership,
    plot_fuzzy_surface, plot_fuzzy_contour,
    plot_time_series_comparison, plot_metrics_table_figure,
    plot_fuzzy_type_comparison, plot_copula_density,
    plot_veg_forecast_with_neighbors
)


def apply_filters(df, crop, greenhouse, varieties):
    """Универсальная функция применения фильтров к DataFrame."""
    df_work = df.copy()
    
    # Фильтр по культуре
    crop_col = next((c for c in ['Тип культуры', 'crop_type'] if c in df.columns), None)
    if crop_col and crop:
        df_work = df_work[df_work[crop_col] == crop]
    
    # Фильтр по теплице
    if greenhouse and greenhouse != 'ALL':
        gh_col = next((c for c in ['ID теплицы', 'greenhouse_id'] if c in df.columns), None)
        if gh_col:
            df_work = df_work[df_work[gh_col].astype(str) == str(greenhouse)]
    
    # Фильтр по сортам
    if varieties:
        var_col = next((c for c in ['Сорт', 'variety'] if c in df.columns), None)
        if var_col:
            df_work = df_work[df_work[var_col].isin(varieties)]
            
    return df_work


def register_callbacks(app):
    """Регистрация всех callback'ов."""
    
    # ==================== ЗАГРУЗКА ФАЙЛА ====================
    @app.callback(
        [Output('upload-status', 'children'),
         Output('data-preview', 'children'),
         Output('data-store', 'data'),
         Output('filter-crop', 'options'),
         Output('filter-crop', 'value'),
         Output('target-dropdown', 'options'),
         Output('target-dropdown', 'value'),
         Output('dataset-info', 'children')],
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        prevent_initial_call=True
    )
    def handle_upload(contents, filename):
        if contents is None:
            return [no_update] * 8
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        ext = os.path.splitext(filename)[1]
        
        temp_path = os.path.join(TEMP_UPLOAD_DIR, f"upload_{int(time.time())}{ext}")
        with open(temp_path, 'wb') as f:
            f.write(decoded)
        
        try:
            df = smart_read_file(temp_path)
            
            file_hash = get_file_hash(filename)
            cache_manager.save_dataframe(file_hash, df)
            
            GLOBAL_MEMORY['file_hash'] = file_hash
            GLOBAL_MEMORY['filename'] = filename
            GLOBAL_MEMORY['results'] = None
            
            preview = dash_table.DataTable(
                data=df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                page_size=5,
            )
            
            info_card = dbc.Card([
                dbc.CardBody([
                    html.H6("📋 Информация о датасете"),
                    html.P(f"Строк: {df.shape[0]} | Столбцов: {df.shape[1]}"),
                    html.P(f"Файл: {filename}"),
                ])
            ], className="mb-3")
            
            status = dbc.Alert(f"✅ Файл '{filename}' загружен!", color="success", dismissable=True)
            
            crop_col = next((c for c in ['Тип культуры', 'crop_type'] if c in df.columns), None)
            crop_opts, crop_val = [], None
            if crop_col:
                unique = df[crop_col].dropna().unique()
                crop_opts = [{'label': str(c), 'value': str(c)} for c in unique]
                crop_val = crop_opts[0]['value'] if crop_opts else None
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_candidates = [c for c in num_cols if c not in EXCLUDE_FROM_FEATURES]
            target_opts = [{'label': c, 'value': c} for c in target_candidates]
            default_target = TARGET_DEFAULT if TARGET_DEFAULT in target_candidates else (
                target_candidates[-1] if target_candidates else None)
            
            return status, preview, {'filename': filename, 'hash': file_hash}, crop_opts, crop_val, target_opts, default_target, info_card
            
        except Exception as e:
            error = dbc.Alert(f"❌ Ошибка: {str(e)}", color="danger")
            return error, "", None, [], None, [], None, ""
    
    # ==================== ФИЛЬТРЫ ====================
    @app.callback(
        [Output('filter-greenhouse', 'options'),
         Output('filter-greenhouse', 'value'),
         Output('filter-variety', 'options'),
         Output('filter-variety', 'value')],
        Input('filter-crop', 'value'),
        prevent_initial_call=True
    )
    def update_crop_filters(crop_value):
        if crop_value is None or GLOBAL_MEMORY.get('file_hash') is None:
            return [], None, [], None
        
        df = cache_manager.load_dataframe(GLOBAL_MEMORY['file_hash'])
        if df is None:
            return [], None, [], None

        crop_col = next((c for c in ['Тип культуры', 'crop_type'] if c in df.columns), None)
        if not crop_col:
            return [], None, [], None
        
        df_filt = df[df[crop_col] == crop_value]
        
        gh_col = next((c for c in ['ID теплицы', 'greenhouse_id'] if c in df.columns), None)
        gh_opts = [{'label': '🏠 Все', 'value': 'ALL'}]
        if gh_col:
            gh_opts += [{'label': str(v), 'value': str(v)} for v in df_filt[gh_col].dropna().unique()]
        
        var_col = next((c for c in ['Сорт', 'variety'] if c in df.columns), None)
        var_opts, var_val = [], None
        if var_col:
            unique = df_filt[var_col].dropna().unique()
            var_opts = [{'label': str(v), 'value': str(v)} for v in unique]
            var_val = [var_opts[0]['value']] if var_opts else None
        
        return gh_opts, 'ALL', var_opts, var_val
    
    # ==================== ПРИЗНАКИ ====================
    @app.callback(
        [Output('features-checklist', 'options'),
         Output('features-checklist', 'value'),
         Output('top-k-slider', 'max'),
         Output('top-k-slider', 'value'),
         Output('correlation-plot', 'figure')],
        [Input('target-dropdown', 'value'),
         Input('top-k-slider', 'value'),
         Input('filter-crop', 'value'),
         Input('filter-greenhouse', 'value'),
         Input('filter-variety', 'value')],
        prevent_initial_call=True
    )
    def update_features(target, top_k, crop, greenhouse, varieties):
        if target is None or GLOBAL_MEMORY.get('file_hash') is None:
            return [], [], 10, 5, go.Figure()
        
        df = cache_manager.load_dataframe(GLOBAL_MEMORY['file_hash'])
        if df is None:
            return [], [], 1, 1, go.Figure()
        
        df_filtered = apply_filters(df, crop, greenhouse, varieties)
        
        if len(df_filtered) < 5:
            empty_fig = go.Figure().add_annotation(
                text="⚠️ Недостаточно данных после применения фильтров", 
                showarrow=False, font=dict(size=14, color="red")
            )
            return [], [], 1, 1, empty_fig
        
        df_clean, _ = preprocess_data(df_filtered, target)
        
        corr_df = compute_correlations(df_clean, target)
        if corr_df.empty:
            return [], [], 1, 1, go.Figure()
        
        max_k = min(len(corr_df), 20)
        actual_k = min(top_k or 5, max_k)
        top = corr_df.head(actual_k)
        
        options = [
            {'label': f"{row['Признак']} (r={row['Корреляция (r)']:.3f})", 
             'value': row['Признак']}
            for _, row in top.iterrows()
        ]
        values = top['Признак'].tolist()
        
        fig = plot_correlation_heatmap(corr_df, target)
        
        return options, values, max_k, actual_k, fig
    
    # ==================== ЗАПУСК АНАЛИЗА (BACKGROUND) ====================
    @app.callback(
        [Output('results-store', 'data'),
         Output('progress-msg', 'children'),
         Output('outdated-banner', 'is_open')],
        Input('run-btn', 'n_clicks'),
        [State('filter-crop', 'value'),
         State('filter-greenhouse', 'value'),
         State('filter-variety', 'value'),
         State('target-dropdown', 'value'),
         State('features-checklist', 'value'),
         State('data-store', 'data')],
        background=True,
        cancel=[Input('cancel-btn', 'n_clicks')],
        running=[
            (Output('run-btn', 'disabled'), True, False),
            (Output('cancel-btn', 'disabled'), False, True),
            (Output('progress-msg', 'children'), '⏳ Выполняется анализ...', ''),
        ],
        prevent_initial_call=True
    )
    def run_analysis(n_clicks, crop, greenhouse, varieties, target, features, data_store):
        if not features or not target:
            return None, "⚠️ Выберите target и признаки!", False
        
        if not data_store or 'hash' not in data_store:
            return None, "❌ Данные не загружены!", False
        
        file_hash = data_store['hash']
        df = cache_manager.load_dataframe(file_hash)
        
        if df is None:
            return None, "❌ Данные не найдены в кэше. Загрузите файл заново!", False
        
        df_work = df.copy()
        df_work = apply_filters(df_work, crop, greenhouse, varieties)
        
        if len(df_work) < 10:
            return None, f"⚠️ Мало данных после фильтрации ({len(df_work)} строк)", False
        
        cache_key = get_analysis_cache_key(file_hash, target, features)
        cached = cache_manager.get(cache_key)
        if cached is not None:
            cache_manager.set("latest_results", cached, expire=86400)
            return {'status': 'cached'}, f"✅ Из кэша ({format_timestamp()})", False
        
        # ETL
        df_clean, report = preprocess_data(df_work, target)
        df_ml = encode_categorical(df_clean)
        
        valid_features = [f for f in features if f in df_ml.columns]
        if not valid_features:
            return None, "❌ Нет валидных признаков после OHE", False
        
        X = df_ml[valid_features].select_dtypes(include=[np.number])
        y = df_ml[target]
        
        valid_mask = X.notnull().all(axis=1) & y.notnull()
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 20:
            return None, f"⚠️ Мало данных ({len(X)} строк)", False
        
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # === МОДЕЛИ ML ===
        ml_results = {}
        for name, func in [('OLS', train_ols), ('Random Forest', train_random_forest), 
                           ('XGBoost', train_xgboost)]:
            try:
                ml_results[name] = func(X_train, y_train, X_test, y_test)
            except Exception as e:
                ml_results[name] = {'error': str(e)}
        
        # === FUZZY TYPE-1 ===
        try:
            fuzzy_data = build_fuzzy_system(X_train, y_train, valid_features)
        except Exception as e:
            fuzzy_data = {'error': str(e)}

        # === FUZZY TYPE-2 ===
        try:
            fuzzy2_data = build_fuzzy_type2_from_data(X_train, y_train, valid_features)
        except Exception as e:
            fuzzy2_data = {'error': str(e)}
            
        # === КОПУЛА ===
        copula_data = None
        if len(valid_features) > 0:
            try:
                top_feat = valid_features[0]
                analyzer = CopulaAnalyzer(y_train, X_train[top_feat])
                copula_data = analyzer.fit_all()
            except Exception as e:
                copula_data = {'error': str(e)}

        # === ВЕГЕТАЦИЯ: обучение ВСЕХ 6 моделей ===
        veg_models_dict = {}
        X_test_veg = None
        y_test_veg = None
        plant_col = next((c for c in ['Дата посадки', 'planting_date'] if c in df_work.columns), None)
        harvest_col = next((c for c in ['Дата сбора урожая', 'harvest_date'] if c in df_work.columns), None)
        
        if plant_col and harvest_col:
            try:
                df_temp = df_work.copy()
                df_temp['planting_dt'] = pd.to_datetime(df_temp[plant_col], errors='coerce')
                df_temp['harvest_dt'] = pd.to_datetime(df_temp[harvest_col], errors='coerce')
                df_temp['duration_days'] = (df_temp['harvest_dt'] - df_temp['planting_dt']).dt.days
                df_temp = df_temp.dropna(subset=['duration_days'])
                df_temp = df_temp[df_temp['duration_days'] > 0]
                
                if len(df_temp) >= 20:
                    X_v = df_ml.loc[df_temp.index][valid_features].select_dtypes(include=[np.number]).fillna(0)
                    y_v = df_temp['duration_days']
                    split_v = int(len(X_v) * 0.8)
                    X_tv, X_tev = X_v.iloc[:split_v], X_v.iloc[split_v:]
                    y_tv, y_tev = y_v.iloc[:split_v], y_v.iloc[split_v:]
                    X_test_veg = X_tev
                    y_test_veg = y_tev
                    
                    # OLS
                    try:
                        X_tv_c = sm.add_constant(X_tv); X_tev_c = sm.add_constant(X_tev)
                        m = sm.OLS(y_tv, X_tv_c).fit()
                        veg_models_dict['OLS'] = m
                    except: pass
                    # RF
                    try:
                        m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_tv, y_tv)
                        veg_models_dict['Random Forest'] = m
                    except: pass
                    # XGB
                    try:
                        m = xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0).fit(X_tv, y_tv)
                        veg_models_dict['XGBoost'] = m
                    except: pass
                    # Fuzzy-1
                    try:
                        fz1 = build_fuzzy_system(X_tv, y_tv, valid_features)
                        veg_models_dict['Fuzzy Type-1'] = fz1['fis']
                    except: pass
                    # Fuzzy-2
                    try:
                        fz2 = build_fuzzy_type2_from_data(X_tv, y_tv, valid_features)
                        veg_models_dict['Fuzzy Type-2'] = fz2['fis']
                    except: pass
                    # Copula Regressor
                    try:
                        m = GaussianCopulaRegressor()
                        m.fit(X_tv.values, y_tv.values)
                        veg_models_dict['Copula'] = m
                    except: pass
            except Exception:
                pass
        
        # === ИТОГОВЫЙ СЛОВАРЬ РЕЗУЛЬТАТОВ (один раз!) ===
        results = {
            'ml': ml_results, 
            'fuzzy': fuzzy_data, 
            'fuzzy2': fuzzy2_data, 
            'copula': copula_data,
            'veg_models': veg_models_dict,
            'X_test_veg': X_test_veg.to_dict() if X_test_veg is not None else None,
            'y_test_veg': y_test_veg.tolist() if y_test_veg is not None else None,
            'report': report, 
            'y_test': y_test.tolist(),
            'X_test': X_test.to_dict(), 
            'features': valid_features,
            'target': target, 
            'train_size': len(X_train), 
            'test_size': len(X_test),
        }
        
        cache_manager.set("latest_results", results, expire=86400)
        cache_manager.set(cache_key, results, expire=86400)
        
        total_time = sum(ml_results[m].get('time', 0) for m in ml_results 
                         if isinstance(ml_results[m], dict) and 'time' in ml_results[m])
        
        return {'status': 'success'}, f"✅ Готово за {total_time:.1f}с", False
    
    # ==================== УСТАРЕВАНИЕ ====================
    @app.callback(
        Output('outdated-banner', 'is_open', allow_duplicate=True),
        [Input('target-dropdown', 'value'),
         Input('features-checklist', 'value'),
         Input('filter-crop', 'value'),
         Input('filter-greenhouse', 'value'),
         Input('filter-variety', 'value')],
        prevent_initial_call=True
    )
    def check_outdated(target, features, crop, greenhouse, varieties):
        results = cache_manager.get("latest_results")
        if results is not None:
            return True
        return False
    
    # ==================== РЕНДЕРИНГ ВКЛАДОК ====================
    @app.callback(
        Output('tab-content', 'children'),
        Input('results-tabs', 'active_tab'),
        prevent_initial_call=False
    )
    def render_tab(active_tab):
        results = cache_manager.get("latest_results")
        if not results:
            return dbc.Alert("📊 Загрузите данные и запустите анализ", color="info")
        
        if active_tab == 'tab-overview':
            return _render_overview(results)
        elif active_tab == 'tab-ml':
            return _render_ml(results)
        elif active_tab == 'tab-fuzzy':
            return _render_fuzzy(results)
        elif active_tab == 'tab-copula':
            return _render_copula(results)
        elif active_tab == 'tab-veg':
            return _render_veg(results)
        return html.P("Выберите вкладку")
    
    # ==================== ПРОГНОЗ ВЕГЕТАЦИИ ====================
    @app.callback(
        Output('veg-result', 'children'),
        Input('veg-predict-btn', 'n_clicks'),
        [State({'type': 'veg-input', 'index': ALL}, 'value'),
         State({'type': 'veg-input', 'index': ALL}, 'id')],
        prevent_initial_call=True
    )
    def predict_veg(n_clicks, values, ids):
        results = cache_manager.get("latest_results")
        if not results:
            return dbc.Alert("❌ Данные не загружены или анализ не запущен", color="danger")
            
        veg_models = results.get('veg_models')
        X_test_veg = results.get('X_test_veg')
        y_test_veg = results.get('y_test_veg')
        
        if not veg_models or X_test_veg is None or y_test_veg is None:
            return dbc.Alert("⚠️ Модели для прогноза вегетации не обучены. Убедитесь, что в файле есть даты посадки и сбора.", color="warning")

        features = [id_dict['index'] for id_dict in ids]
        input_dict = {feat: float(val) if val is not None else 0.0 for feat, val in zip(features, values)}
        input_vector = np.array([input_dict[f] for f in features])
        X_test_df = pd.DataFrame(X_test_veg)
        y_test_ser = pd.Series(y_test_veg)

        elements = []
        k = 5

        for model_name, model in veg_models.items():
            try:
                if 'Fuzzy' in model_name:
                    pred = float(model.compute(input_dict))
                else:
                    df_input = pd.DataFrame([input_dict])
                    for col in X_test_df.columns:
                        if col not in df_input.columns:
                            df_input[col] = 0
                    df_input = df_input[X_test_df.columns]
                    pred = float(model.predict(df_input)[0])

                X_test_vals = X_test_df[features].values
                dists = np.linalg.norm(X_test_vals - input_vector.reshape(1, -1), axis=1)
                nearest_idx = np.argsort(dists)[:k]
                nearest_y = y_test_ser.iloc[nearest_idx].values

                fig = plot_veg_forecast_with_neighbors(model_name, pred, X_test_df[features], y_test_ser, input_vector, k)
                elements.append(
                    dbc.Card([dbc.CardBody([dcc.Graph(figure=fig)])], className="mb-4 shadow-sm")
                )
            except Exception as e:
                elements.append(
                    dbc.Alert(f"⚠️ Ошибка в модели {model_name}: {str(e)}", color="warning", className="mb-2")
                )

        return html.Div(elements)


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ РЕНДЕРИНГА ====================

def _render_overview(results):
    report_items = [html.Li(r) for r in results.get('report', [])]
    
    veg_ok = results.get('veg_models') and len(results.get('veg_models', {})) > 0
    fuzzy_ok = results.get('fuzzy') and 'error' not in results.get('fuzzy', {})
    fuzzy2_ok = results.get('fuzzy2') and 'error' not in results.get('fuzzy2', {})
    copula_ok = results.get('copula') and 'error' not in results.get('copula', {})
    
    table_data = [
        {'Метод': 'OLS', 'Доступен': '✅'},
        {'Метод': 'Random Forest', 'Доступен': '✅'},
        {'Метод': 'XGBoost', 'Доступен': '✅'},
        {'Метод': 'Fuzzy Type-1', 'Доступен': '✅' if fuzzy_ok else '❌'},
        {'Метод': 'Fuzzy Type-2', 'Доступен': '✅' if fuzzy2_ok else '❌'},
        {'Метод': 'Копула', 'Доступен': '✅' if copula_ok else '❌'},
        {'Метод': 'Прогноз вегетации', 'Доступен': '✅' if veg_ok else '❌'},
    ]
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader("📝 Отчёт ETL"),
            dbc.CardBody([
                html.Ul(report_items),
                html.P(f"Train: {results.get('train_size')} | Test: {results.get('test_size')}"),
            ])
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader("📋 Применимость методов"),
            dbc.CardBody([
                dash_table.DataTable(data=table_data, 
                                     columns=[{"name": i, "id": i} for i in ['Метод', 'Доступен']])
            ])
        ])
    ])


def _render_ml(results):
    ml = results.get('ml', {})
    y_test = results.get('y_test', [])
    
    elements = [dcc.Graph(figure=plot_metrics_table_figure(ml))]
    elements.append(dbc.Card([dbc.CardBody([dcc.Graph(figure=plot_models_comparison(ml))])], className="mb-3"))
    
    comp = []
    for name in ['Random Forest', 'XGBoost']:
        if name in ml and 'error' not in ml[name]:
            comp.append(dbc.Col(dcc.Graph(figure=plot_actual_vs_predicted(
                y_test, ml[name]['pred'], name, ml[name]['r2'])), md=6))
    if comp:
        elements.append(dbc.Row(comp, className="mb-3"))
    
    imp = []
    for name in ['Random Forest', 'XGBoost']:
        if name in ml and 'importances' in ml[name]:
            imp.append(dbc.Col(dcc.Graph(figure=plot_feature_importance(
                ml[name]['importances'], name)), md=6))
    if imp:
        elements.append(dbc.Row(imp, className="mb-3"))
    
    if 'OLS' in ml and 'error' not in ml['OLS']:
        elements.append(dbc.Card([
            dbc.CardBody([dcc.Graph(figure=plot_ols_diagnostics(ml['OLS']))])
        ], className="mb-3"))
    
    if 'Random Forest' in ml and 'XGBoost' in ml:
        rf_p = ml['Random Forest'].get('pred', [])
        xgb_p = ml['XGBoost'].get('pred', [])
        if len(rf_p) == len(y_test) and len(xgb_p) == len(y_test):
            elements.append(dbc.Card([
                dbc.CardBody([dcc.Graph(figure=plot_time_series_comparison(y_test, rf_p, xgb_p))])
            ], className="mb-3"))
    
    return html.Div(elements)


def _render_fuzzy(results):
    fuzzy = results.get('fuzzy')
    fuzzy2 = results.get('fuzzy2')
    
    if not fuzzy or 'error' in fuzzy:
        return dbc.Alert(f"⚠️ Fuzzy Type-1: {fuzzy.get('error', 'не построена')}", color="warning")
    
    elements = [
        dbc.Alert(
            f"🧠 Type-1 правил: {fuzzy.get('rules_count')} | "
            f"Type-2 правил: {fuzzy2.get('rules_count') if fuzzy2 and 'error' not in fuzzy2 else 'N/A'} | "
            f"Признаков: {len(fuzzy.get('features', []))}",
            color="info", className="mb-3"
        ),
        dbc.Card([
            dbc.CardHeader("📐 Функции принадлежности (Type-1)"),
            dbc.CardBody([dcc.Graph(figure=plot_fuzzy_membership(fuzzy))])
        ], className="mb-3")
    ]
    
    X_test = pd.DataFrame(results.get('X_test', {}))
    if fuzzy2 and 'error' not in fuzzy2 and len(fuzzy.get('features', [])) >= 2 and not X_test.empty:
        elements.append(dbc.Card([
            dbc.CardHeader("📊 Сравнение Type-1 vs Type-2"),
            dbc.CardBody([dcc.Graph(figure=plot_fuzzy_type_comparison(fuzzy, fuzzy2, X_test))])
        ], className="mb-3"))
    elif len(fuzzy.get('features', [])) >= 2 and not X_test.empty:
        elements.append(dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(figure=plot_fuzzy_surface(fuzzy, X_test))])]), md=6),
            dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(figure=plot_fuzzy_contour(fuzzy, X_test))])]), md=6),
        ], className="mb-3"))
    
    return html.Div(elements)


def _render_copula(results):
    copula = results.get('copula')
    if not copula or 'error' in copula:
        return dbc.Alert("⚠️ Копула не построена", color="warning")

    table_data = []
    for name, res in copula.items():
        param_str = f"{res['param']:.3f}" if isinstance(res['param'], float) else f"ρ={res['param'][0]:.2f}, ν={res['param'][1]:.1f}"
        table_data.append({
            'Копула': name, 
            'Параметр': param_str, 
            'Log-Lik': f"{res['loglik']:.2f}",
            'AIC': f"{res['aic']:.2f}", 
            'BIC': f"{res['bic']:.2f}",
            'Левый хвост (λL)': f"{res['tail_l']:.3f}", 
            'Правый хвост (λU)': f"{res['tail_u']:.3f}"
        })

    best_copula_name = min(copula.items(), key=lambda x: x[1]['aic'])[0]
    best_rho = copula[best_copula_name]['param']
    if isinstance(best_rho, tuple): 
        best_rho = best_rho[0]
        
    copula_fig = plot_copula_density(best_rho, best_copula_name)

    return html.Div([
        dbc.Card([
            dbc.CardHeader("📊 Анализ зависимостей (Копулы)"),
            dbc.CardBody([
                html.H5("Метрики качества копул (на паре Target - Top Feature)"),
                dash_table.DataTable(
                    data=table_data, 
                    columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'fontSize': 12},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'}
                ),
                html.Hr(),
                html.H5(f"Графическое представление плотности лучшей копулы ({best_copula_name})"),
                dcc.Graph(figure=copula_fig)
            ])
        ])
    ])


def _render_veg(results):
    veg_models = results.get('veg_models')
    if not veg_models:
        return dbc.Alert("🌱 Модели вегетации не обучены. Убедитесь, что в файле есть столбцы 'Дата посадки' и 'Дата сбора урожая'.", color="warning")
    
    features = results.get('features', [])
    X_test = pd.DataFrame(results.get('X_test', {}))
    mean_vals = {f: float(X_test[f].mean()) for f in features if f in X_test.columns}
    
    inputs = []
    for feat in features:
        inputs.append(dbc.Row([
            dbc.Col(html.Label(feat, className="fw-bold"), md=5),
            dbc.Col(dcc.Input(id={'type': 'veg-input', 'index': feat},
                              type='number', value=round(mean_vals.get(feat, 0), 2),
                              step=0.1, className="form-control"), md=7),
        ], className="mb-2"))
    
    return dbc.Card([
        dbc.CardHeader("🌱 Прогноз вегетации (мульти-модельный)"),
        dbc.CardBody([
            dbc.Alert(f"Обучено моделей: {len(veg_models)} | Признаков: {len(features)}", color="info"),
            html.Hr(),
            html.Div(inputs),
            dbc.Button("🔮 Спрогнозировать", id="veg-predict-btn", color="success", className="mt-3"),
            html.Div(id="veg-result", className="mt-3"),
        ])
    ])