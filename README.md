# Проект — Dash/ML приложение

Краткое описание
- Это локальное веб-приложение на основе Dash/Flask для анализа и визуализации данных, содержащих модули для предобработки, модели, Fuzzy/копула движок и визуализацию.

Требования
- Python 3.10+ (рекомендуется)
- Все зависимости перечислены в `requirements.txt`

Установка
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# или Windows CMD
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Запуск
```bash
# После активации виртуального окружения
python app.py
```
Откройте в браузере http://127.0.0.1:8050 (или адрес/порт, указанный в логах).

Тестирование
```bash
# Запустить все unit-тесты
python -m pytest

# Запустить конкретный файл тестов
python -m pytest tests/test_data_processing.py
```

Покрытие
```bash
# Установите pytest-cov, если ещё не установлено
pip install pytest-cov

# Запустить тесты с отчётом по покрытию
python -m pytest --cov=./ --cov-report=term-missing
```

Краткая структура репозитория
- `app.py`: основной запуск приложения (Dash/Flask)
- `requirements.txt`: зависимости Python
- `cache_manager.py`: управление кэшем
- `callbacks.py`: колбэки Dash
- `config.py`: конфигурация приложения
- `copula_engine.py`, `fuzzy_engine.py`: математические/логические модули
- `data_processing.py`: предобработка и загрузка данных
- `models.py`: модели и логика машинного обучения
- `visualizations.py`: функции построения графиков
- `utils.py`: вспомогательные утилиты
- `assets/`: статические файлы (например, `style.css`)
- `dash_cache/`: кэш Dash-приложения (может быть очищен при необходимости)
- `temp_uploads/`: временные загруженные файлы (можно очищать вручную)

Советы и примечания
- Если вы используете Windows PowerShell и получаете ошибку при активации, выполните:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```
- В случае ошибок посмотрите логи в консоли при запуске `app.py`.

Как внести изменения / вклад
- Создавайте ветки для фич и багфиксов, оформляйте PR с описанием изменений.
- Добавляйте тесты для новой логики (если применимо).

Контакты
- По вопросам работы проекта напишите автору/поддержке (добавьте свои контакты здесь).

---
Ниже — расширенные инструкции по основным компонентам и примерные сценарии использования.

## Детальное описание компонентов

- `app.py`: основной Dash-приложение — содержит layout, инициализацию `DiskcacheManager` и точку входа для локального запуска. Параметры хоста/порта и флаг `DEBUG` настраиваются в `config.py`.
- `data_processing.py`: функции для импорта (`smart_read_file`), обзора данных (`get_data_overview`), предобработки (`preprocess_data`), кодирования категорий (`encode_categorical`) и вычисления корреляций (`compute_correlations`). Используйте эти функции для ETL/EDA шагов перед обучением моделей.
- `models.py`: набор обучающих функций — `train_ols`, `train_random_forest`, `train_xgboost`, `train_copula_regressor`, вспомогательные обёртки для нечётких систем (`build_fuzzy_system`, `fuzzy_predict`) и модели для прогноза длительности вегетации.
- `fuzzy_engine.py`: движок для построения нечётких систем (Type-1 / Type-2). Интерфейс используется через обёртки в `models.py` и callbacks.
- `copula_engine.py`: инструменты для анализа зависимостей через копулы и регрессии на их основе; визуализация плотностей интегрирована в `visualizations.py`.
- `visualizations.py`: функции генерации графиков Plotly (корреляции, факт/прогноз, важности признаков, диагностика OLS, графики Fuzzy и плотности копул).
- `callbacks.py`: glue-код Dash — обработка загрузки, фильтров, запуска анализа в background, сбор и кэширование результатов, рендеринг вкладок (`ML`, `Fuzzy`, `Copula`, `Veg`).
- `cache_manager.py`: локальный кэширующий слой (используется при загрузке файлов и сохранении результатов анализа).
- `config.py`: все настраиваемые параметры (папки для загрузок/кэша, пороги корреляции/пропусков, параметры CV/поиска гиперпараметров и т.д.).

## Быстрые примеры использования (локально, в REPL или скриптах)

- Прочитать файл и получить обзор данных:

```python
from data_processing import smart_read_file, get_data_overview

df = smart_read_file('path/to/my_dataset.csv')
overview = get_data_overview(df)
print(overview['shape'], overview['columns'][:10])
```

- Предобработать данные и получить корреляции с целевой колонкой:

```python
from data_processing import preprocess_data, compute_correlations

df_clean, report = preprocess_data(df, target_col='yield')
print('\n'.join(report))
corr_df = compute_correlations(df_clean, 'yield')
print(corr_df.head())
```

- Обучить модель Random Forest (локально, вне Dash):

```python
from models import train_random_forest
from sklearn.model_selection import train_test_split

X = df_clean.select_dtypes(include=['number']).drop(columns=['yield'])
y = df_clean['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
res = train_random_forest(X_train, y_train, X_test, y_test)
print(res['r2'], res.get('best_params'))
```

- Построить график факт vs прогноз и показать в отдельном окне (Plotly):

```python
from visualizations import plot_actual_vs_predicted
fig = plot_actual_vs_predicted(y_test, res['pred'], 'Random Forest', res['r2'])
fig.show()
```

- Построить нечёткую систему и посмотреть функции принадлежности:

```python
from models import build_fuzzy_system
fuzzy = build_fuzzy_system(X_train, y_train, X_train.columns.tolist())
# В Dash это интегрировано в вкладку Fuzzy; вручную можно визуализировать через visualizations.plot_fuzzy_membership
```

## Полезные команды и советы

- Запуск приложения (в виртуальном окружении):

```bash
python app.py
```

- Если нужно очистить кэш Dash / временные загрузки, можно удалить содержимое папок `dash_cache/` и `temp_uploads/`.
- Настройки по умолчанию и пороги находятся в `config.py`. Для воспроизводимости фиксируйте `RANDOM_STATE`.

## Зависимости
- Все зависимости в `requirements.txt`. В проекте используются: `dash`, `dash-bootstrap-components`, `diskcache`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `statsmodels`, `plotly`, `scipy` и др.

Если хотите, могу: добавить интерактивные GIF/скриншоты интерфейса, детализировать API отдельных функций или добавить пошаговый пример с реальным CSV-файлом из `temp_uploads/`.