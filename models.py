# models.py
"""
Модуль обучения моделей: OLS, Random Forest, XGBoost, Fuzzy Logic.
"""

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from copula_engine import CopulaAnalyzer, GaussianCopulaRegressor
from fuzzy_engine import build_fuzzy_type2_from_data

from config import (
    RF_PARAM_GRID, RF_N_ITER, XGB_PARAM_GRID, XGB_N_ITER,
    CV_SPLITS, RANDOM_STATE, FUZZY_MAX_FEATURES
)
from utils import calculate_mape
from fuzzy_engine import build_fuzzy_from_data


def train_ols(X_train, y_train, X_test, y_test) -> dict:
    """OLS регрессия с диагностикой."""
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)
    
    t0 = time.time()
    try:
        model = sm.OLS(y_train, X_train_c).fit()
        y_pred = model.predict(X_test_c)
        train_time = time.time() - t0
        
        return {
            'model': model,
            'pred': y_pred,
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': calculate_mape(y_test, y_pred),
            'time': train_time,
            'summary': model.summary().as_text(),
            'residuals': y_test.values - y_pred,
            'fitted': y_pred,
        }
    except Exception as e:
        return {'error': str(e), 'time': time.time() - t0}


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Random Forest с RandomizedSearchCV."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=RF_PARAM_GRID,
        n_iter=RF_N_ITER,
        cv=tscv,
        scoring='r2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    t0 = time.time()
    rf_search.fit(X_train, y_train)
    train_time = time.time() - t0
    
    best_model = rf_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    importances = dict(zip(X_train.columns, best_model.feature_importances_))
    
    return {
        'model': best_model,
        'best_params': rf_search.best_params_,
        'pred': y_pred,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mape': calculate_mape(y_test, y_pred),
        'time': train_time,
        'importances': importances,
    }


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """XGBoost с регуляризацией."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    
    xgb_base = xgb.XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
        verbosity=0
    )
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=XGB_PARAM_GRID,
        n_iter=XGB_N_ITER,
        cv=tscv,
        scoring='r2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    t0 = time.time()
    xgb_search.fit(X_train, y_train)
    train_time = time.time() - t0
    
    best_model = xgb_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    importances = dict(zip(X_train.columns, best_model.feature_importances_))
    
    return {
        'model': best_model,
        'best_params': xgb_search.best_params_,
        'pred': y_pred,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mape': calculate_mape(y_test, y_pred),
        'time': train_time,
        'importances': importances,
    }


def train_vegetation_ols(X_train, y_train, X_test, y_test):
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train_c).fit()
    y_pred = model.predict(X_test_c)
    return {
        'model': model, 'pred': y_pred,
        'r2': r2_score(y_test, y_pred), 'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

def train_copula_regressor(X_train, y_train, X_test, y_test):
    # Используем только топ-1 признак для бивариатной копулы, или Гауссову для многомерной
    # Для простоты и скорости в вебе используем GaussianCopulaRegressor из нашего движка
    model = GaussianCopulaRegressor()
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    # Ограничиваем прогноз диапазоном y_train
    y_pred = np.clip(y_pred, y_train.min(), y_train.max())
    return {
        'model': model, 'pred': y_pred,
        'r2': r2_score(y_test, y_pred), 'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }


def build_fuzzy_system(X_train: pd.DataFrame, y_train: pd.Series, 
                       feature_names: list) -> dict:
    """Обёртка для построения нечёткой системы."""
    try:
        return build_fuzzy_from_data(X_train, y_train, feature_names, FUZZY_MAX_FEATURES)
    except Exception as e:
        return {'error': f'Ошибка построения Fuzzy: {str(e)}'}


def fuzzy_predict(fuzzy_data: dict, input_values: dict) -> float:
    """Прогноз через нечёткую систему."""
    if 'error' in fuzzy_data:
        return np.nan
    try:
        return fuzzy_data['fis'].compute(input_values)
    except Exception:
        return np.nan


def train_vegetation_model(df: pd.DataFrame, features: list,
                           date_col_plant: str, date_col_harvest: str) -> dict:
    """Модель прогноза длительности вегетации."""
    df_temp = df.copy()
    
    df_temp['planting_dt'] = pd.to_datetime(df_temp[date_col_plant], errors='coerce')
    df_temp['harvest_dt'] = pd.to_datetime(df_temp[date_col_harvest], errors='coerce')
    df_temp['duration_days'] = (df_temp['harvest_dt'] - df_temp['planting_dt']).dt.days
    
    df_temp = df_temp.dropna(subset=['duration_days'])
    df_temp = df_temp[df_temp['duration_days'] > 0]
    
    if len(df_temp) < 10:
        return {'error': 'Недостаточно данных для прогноза вегетации'}
    
    valid_features = [f for f in features if f in df_temp.columns]
    X = df_temp[valid_features].select_dtypes(include=[np.number])
    y = df_temp['duration_days']
    
    if X.shape[1] == 0:
        return {'error': 'Нет числовых признаков для модели вегетации'}
    
    X = X.fillna(X.median())
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    model.fit(X, y)
    
    return {
        'model': model,
        'features': X.columns.tolist(),
        'mean_values': X.mean().to_dict(),
        'train_size': len(X),
        'r2': r2_score(y, model.predict(X)),
    }