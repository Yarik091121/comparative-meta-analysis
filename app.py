# # app.py
# """
# Главный файл приложения.
# Исправлен для Dash 2.17+ (DiskcacheManager).
# """

# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# import diskcache
# from dash import DiskcacheManager

# from config import CACHE_DIR, HOST, PORT, DEBUG
# from callbacks import register_callbacks

# # ==================== BACKGROUND MANAGER ====================
# cache = diskcache.Cache(CACHE_DIR)
# background_manager = DiskcacheManager(cache)

# # ==================== APP ====================
# app = dash.Dash(
#     __name__,
#     external_stylesheets=[dbc.themes.FLATLY],
#     suppress_callback_exceptions=True,
#     background_callback_manager=background_manager,
# )
# app.title = "🌿 Мета-анализ тепличного хозяйства"
# server = app.server

# # ==================== LAYOUT ====================
# app.layout = dbc.Container([
#     dcc.Store(id='data-store'),
#     dcc.Store(id='results-store'),
    
#     dbc.Row(dbc.Col(html.Div([
#         html.H1("🌿 Сравнительный мета-анализ тепличного хозяйства",
#                 className="text-center text-success mb-2"),
#         html.P("EDA, ML-моделирование и нечёткая логика",
#                className="text-center text-muted"),
#     ], className="my-4"))),
    
#     # Загрузка
#     dbc.Card([
#         dbc.CardHeader(html.H5("📂 Шаг 1: Загрузка данных")),
#         dbc.CardBody([
#             dcc.Upload(
#                 id='upload-data',
#                 children=html.Div(['Перетащите CSV/Excel или ', html.A('выберите файл')]),
#                 style={'width': '100%', 'height': '70px', 'lineHeight': '70px',
#                        'borderWidth': '2px', 'borderStyle': 'dashed',
#                        'borderRadius': '10px', 'textAlign': 'center',
#                        'backgroundColor': '#f8f9fa', 'cursor': 'pointer'},
#             ),
#             html.Div(id='upload-status', className="mt-3"),
#             html.Div(id='dataset-info', className="mt-2"),
#             html.Div(id='data-preview', className="mt-3"),
#         ])
#     ], className="mb-4 shadow-sm"),
    
#     # Параметры
#     dbc.Card([
#         dbc.CardHeader(html.H5("⚙️ Шаг 2: Параметры")),
#         dbc.CardBody([
#             dbc.Row([
#                 dbc.Col([html.Label("🌾 Культура:"), dcc.Dropdown(id='filter-crop')], md=3),
#                 dbc.Col([html.Label("🏠 Теплица:"), dcc.Dropdown(id='filter-greenhouse')], md=3),
#                 dbc.Col([html.Label("🧬 Сорт:"), dcc.Dropdown(id='filter-variety', multi=True)], md=6),
#             ]),
#             html.Hr(),
#             dbc.Row([
#                 dbc.Col([html.Label("🎯 Target:"), dcc.Dropdown(id='target-dropdown')], md=6),
#                 dbc.Col([html.Label("📊 Топ-K:"),
#                          dcc.Slider(id='top-k-slider', min=1, max=15, step=1, value=5,
#                                     marks={i: str(i) for i in range(1, 16, 2)},
#                                     tooltip={"placement": "bottom", "always_visible": True})], md=6),
#             ], className="mb-3"),
#             dbc.Card([
#                 dbc.CardHeader("✅ Признаки для анализа (отметьте нужные):"),
#                 dbc.CardBody(
#                     dbc.Checklist(
#                         id='features-checklist',
#                         options=[],
#                         value=[],
#                         inline=False,
#                         className="mt-2"
#                     )
#                 ),
#             ], className="mb-3"),
#             dbc.Card([
#                 dbc.CardHeader("📈 Корреляции:"),
#                 dbc.CardBody(dcc.Graph(id='correlation-plot', figure={}))
#             ], className="mb-3"),
#             html.Hr(),
#             dbc.Row([
#                 dbc.Col(dbc.Button("🚀 Запустить анализ", id="run-btn", color="success", size="lg"), md=4),
#                 dbc.Col(dbc.Button("❌ Отменить", id="cancel-btn", color="danger", outline=True, size="lg"), md=3),
#                 dbc.Col(html.Div(id="progress-msg", className="text-info fw-bold mt-2"), md=5),
#             ]),
#         ])
#     ], className="mb-4 shadow-sm"),
    
#     dbc.Alert("⚠️ Параметры изменены!", id="outdated-banner", color="warning", is_open=False, className="mb-3"),
    
#     # Результаты
#     dbc.Card([
#         dbc.CardHeader(html.H5("📊 Результаты")),
#         dbc.CardBody([
#             dcc.Loading(id="loading-results", type="circle", children=html.Div([
#                 dbc.Tabs(id="results-tabs", active_tab="tab-overview", children=[
#                     dbc.Tab(label="📋 Обзор", tab_id="tab-overview"),
#                     dbc.Tab(label="🤖 ML", tab_id="tab-ml"),
#                     dbc.Tab(label="🧠 Fuzzy", tab_id="tab-fuzzy"),
#                     dbc.Tab(label="🌱 Вегетация", tab_id="tab-veg"),
#                 ]),
#                 html.Div(id="tab-content"),
#             ]))
#         ])
#     ], className="mb-4 shadow-sm"),
# ], fluid=True, className="p-4")


# # ==================== РЕГИСТРАЦИЯ ====================
# register_callbacks(app)


# # ==================== ЗАПУСК ====================
# if __name__ == '__main__':
#     print(f"\n{'='*60}")
#     print(f"  🌿 http://{HOST}:{PORT}")
#     print(f"{'='*60}\n")
#     app.run(debug=DEBUG, host=HOST, port=PORT)

# app.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import diskcache
from dash import DiskcacheManager

from config import CACHE_DIR, HOST, PORT, DEBUG
from callbacks import register_callbacks

cache = diskcache.Cache(CACHE_DIR)
background_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    background_callback_manager=background_manager,
)
app.title = "🌿 Мета-анализ тепличного хозяйства"
server = app.server

app.layout = dbc.Container([
    dcc.Store(id='data-store'),
    dcc.Store(id='results-store'),
    
    dbc.Row(dbc.Col(html.Div([
        html.H1("🌿 Сравнительный мета-анализ тепличного хозяйства",
                className="text-center text-success mb-2"),
        html.P("EDA, ML-моделирование, нечёткая логика и копулы",
               className="text-center text-muted"),
    ], className="my-4"))),
    
    dbc.Card([
        dbc.CardHeader(html.H5("📂 Шаг 1: Загрузка данных")),
        dbc.CardBody([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Перетащите CSV/Excel или ', html.A('выберите файл')]),
                style={'width': '100%', 'height': '70px', 'lineHeight': '70px',
                       'borderWidth': '2px', 'borderStyle': 'dashed',
                       'borderRadius': '10px', 'textAlign': 'center',
                       'backgroundColor': '#f8f9fa', 'cursor': 'pointer'},
            ),
            html.Div(id='upload-status', className="mt-3"),
            html.Div(id='dataset-info', className="mt-2"),
            html.Div(id='data-preview', className="mt-3"),
        ])
    ], className="mb-4 shadow-sm"),
    
    dbc.Card([
        dbc.CardHeader(html.H5("⚙️ Шаг 2: Параметры")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("🌾 Культура:"), dcc.Dropdown(id='filter-crop')], md=3),
                dbc.Col([html.Label("🏠 Теплица:"), dcc.Dropdown(id='filter-greenhouse')], md=3),
                dbc.Col([html.Label("🧬 Сорт:"), dcc.Dropdown(id='filter-variety', multi=True)], md=6),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([html.Label("🎯 Target:"), dcc.Dropdown(id='target-dropdown')], md=6),
                dbc.Col([html.Label("📊 Топ-K:"),
                         dcc.Slider(id='top-k-slider', min=1, max=15, step=1, value=5,
                                    marks={i: str(i) for i in range(1, 16, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True})], md=6),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("✅ Признаки для анализа (отметьте нужные):"),
                dbc.CardBody(
                    dbc.Checklist(id='features-checklist', options=[], value=[], inline=False, className="mt-2")
                ),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("📈 Корреляции:"),
                dbc.CardBody(dcc.Graph(id='correlation-plot', figure={}))
            ], className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col(dbc.Button("🚀 Запустить анализ", id="run-btn", color="success", size="lg"), md=4),
                dbc.Col(dbc.Button("❌ Отменить", id="cancel-btn", color="danger", outline=True, size="lg"), md=3),
                dbc.Col(html.Div(id="progress-msg", className="text-info fw-bold mt-2"), md=5),
            ]),
        ])
    ], className="mb-4 shadow-sm"),
    
    dbc.Alert("⚠️ Параметры изменены!", id="outdated-banner", color="warning", is_open=False, className="mb-3"),
    
    dbc.Card([
        dbc.CardHeader(html.H5("📊 Результаты")),
        dbc.CardBody([
            dcc.Loading(id="loading-results", type="circle", children=html.Div([
                dbc.Tabs(id="results-tabs", active_tab="tab-overview", children=[
                    dbc.Tab(label="📋 Обзор", tab_id="tab-overview"),
                    dbc.Tab(label="🤖 ML", tab_id="tab-ml"),
                    dbc.Tab(label="🧠 Fuzzy", tab_id="tab-fuzzy"),
                    dbc.Tab(label="🔗 Копула", tab_id="tab-copula"),  # ✅ НОВАЯ ВКЛАДКА
                    dbc.Tab(label="🌱 Вегетация", tab_id="tab-veg"),
                ]),
                html.Div(id="tab-content"),
            ]))
        ])
    ], className="mb-4 shadow-sm"),
], fluid=True, className="p-4")

register_callbacks(app)

if __name__ == '__main__':
    print(f"\n{'='*60}\n  🌿 http://{HOST}:{PORT}\n{'='*60}\n")
    app.run(debug=DEBUG, host=HOST, port=PORT)