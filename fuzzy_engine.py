# fuzzy_engine.py
"""
Собственный движок нечёткой логики (Mamdani FIS).
Заменяет scikit-fuzzy для совместимости с Python 3.12+.

Поддерживает:
- Треугольные функции принадлежности
- Логика AND через min
- Агрегация через max
- Дефаззификация через centroid
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Optional


def trimf(universe: np.ndarray, params: List[float]) -> np.ndarray:
    """
    Треугольная функция принадлежности.
    params = [a, b, c] — левая ножка, вершина, правая ножка.
    """
    a, b, c = params
    x = universe
    y = np.zeros_like(x, dtype=float)
    
    # Левая сторона
    if b != a:
        idx = np.logical_and(a < x, x < b)
        y[idx] = (x[idx] - a) / (b - a)
    
    # Вершина
    y[x == b] = 1.0
    
    # Правая сторона
    if c != b:
        idx = np.logical_and(b < x, x < c)
        y[idx] = (c - x[idx]) / (c - b)
    
    # За пределами [a, c] = 0
    return y


def fuzzify(value: float, universe: np.ndarray, mf: np.ndarray) -> float:
    """Вычисляет степень принадлежности значения к функции принадлежности."""
    if value <= universe[0]:
        return float(mf[0])
    if value >= universe[-1]:
        return float(mf[-1])
    return float(np.interp(value, universe, mf))


def defuzzify_centroid(universe: np.ndarray, aggregated_mf: np.ndarray) -> float:
    """Дефаззификация методом центроида."""
    total_area = np.sum(aggregated_mf)
    if total_area == 0:
        return float(np.mean(universe))
    return float(np.sum(universe * aggregated_mf) / total_area)


class FuzzyVariable:
    """Нечёткая переменная (антецедент или консеквент)."""
    
    def __init__(self, universe: np.ndarray, name: str):
        self.universe = universe
        self.name = name
        self.terms: Dict[str, np.ndarray] = {}
    
    def add_term(self, name: str, params: List[float]):
        """Добавляет треугольный терм."""
        self.terms[name] = trimf(self.universe, params)


class FuzzyRule:
    """Правило нечёткой логики: IF (условия) THEN (выход)."""
    
    def __init__(self, conditions: Dict[str, str], output_term: str):
        """
        conditions: {имя_переменной: имя_терма}
        output_term: имя терма выхода
        """
        self.conditions = conditions
        self.output_term = output_term


class MamdaniFIS:
    """
    Нечёткая система вывода Мамдани.
    Полная замена skfuzzy.control.ControlSystem.
    """
    
    def __init__(self):
        self.antecedents: Dict[str, FuzzyVariable] = {}
        self.consequent: Optional[FuzzyVariable] = None
        self.rules: List[FuzzyRule] = []
    
    def add_antecedent(self, name: str, universe: np.ndarray) -> FuzzyVariable:
        var = FuzzyVariable(universe, name)
        self.antecedents[name] = var
        return var
    
    def add_consequent(self, name: str, universe: np.ndarray) -> FuzzyVariable:
        self.consequent = FuzzyVariable(universe, name)
        return self.consequent
    
    def add_rule(self, conditions: Dict[str, str], output_term: str):
        self.rules.append(FuzzyRule(conditions, output_term))
    
    def compute(self, inputs: Dict[str, float]) -> float:
        """
        Вычисляет выход системы для заданных входов.
        inputs: {имя_антецедента: числовое_значение}
        """
        if self.consequent is None:
            raise ValueError("Консеквент не задан")
        
        # Агрегированная функция принадлежности выхода
        aggregated = np.zeros_like(self.consequent.universe, dtype=float)
        
        for rule in self.rules:
            # Вычисляем силу срабатывания правила (AND = min)
            firing_strength = 1.0
            for var_name, term_name in rule.conditions.items():
                if var_name not in inputs:
                    firing_strength = 0.0
                    break
                if var_name not in self.antecedents:
                    firing_strength = 0.0
                    break
                
                var = self.antecedents[var_name]
                if term_name not in var.terms:
                    firing_strength = 0.0
                    break
                
                # Фаззификация входа
                membership = fuzzify(inputs[var_name], var.universe, var.terms[term_name])
                firing_strength = min(firing_strength, membership)
            
            if firing_strength > 0:
                # Обрезаем выходной терм (Mamdani implication = min)
                if rule.output_term in self.consequent.terms:
                    clipped = np.minimum(
                        self.consequent.terms[rule.output_term],
                        firing_strength
                    )
                    aggregated = np.maximum(aggregated, clipped)
        
        # Дефаззификация
        return defuzzify_centroid(self.consequent.universe, aggregated)
    
    def compute_grid(self, inputs_grid: Dict[str, np.ndarray], 
                     fixed_inputs: Dict[str, float]) -> np.ndarray:
        """
        Вычисляет выход для сетки значений (для 3D-графиков).
        inputs_grid: {имя: массив_значений} — максимум 2 переменные
        fixed_inputs: {имя: фиксированное_значение} — остальные переменные
        """
        grid_vars = list(inputs_grid.keys())
        if len(grid_vars) == 1:
            x1 = inputs_grid[grid_vars[0]]
            z = np.zeros(len(x1))
            for i, v1 in enumerate(x1):
                inp = {**fixed_inputs, grid_vars[0]: v1}
                z[i] = self.compute(inp)
            return z
        
        elif len(grid_vars) == 2:
            x1 = inputs_grid[grid_vars[0]]
            x2 = inputs_grid[grid_vars[1]]
            X1, X2 = np.meshgrid(x1, x2)
            Z = np.zeros_like(X1)
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    inp = {
                        **fixed_inputs,
                        grid_vars[0]: X1[i, j],
                        grid_vars[1]: X2[i, j]
                    }
                    try:
                        Z[i, j] = self.compute(inp)
                    except Exception:
                        Z[i, j] = np.nan
            return X1, X2, Z
        
        raise ValueError("Поддерживается максимум 2 переменные для сетки")


def build_fuzzy_from_data(X_train, y_train, feature_names: List[str], 
                          max_features: int = 3) -> dict:
    """
    Строит нечёткую систему на основе данных.
    
    Возвращает словарь с:
    - 'fis': объект MamdaniFIS
    - 'features': список использованных признаков
    - 'rules_count': количество правил
    - 'antecedents': словарь FuzzyVariable для визуализации
    - 'consequent': FuzzyVariable выхода
    """
    import pandas as pd
    
    # Ограничиваем количество признаков
    feats = feature_names[:max_features]
    X_sub = X_train[feats].copy()
    
    # Создаём FIS
    fis = MamdaniFIS()
    term_labels = ['низкий', 'средний', 'высокий']
    
    # === Антецеденты ===
    antecedents_dict = {}
    for f in feats:
        mn, mx = float(X_sub[f].min()), float(X_sub[f].max())
        if mn == mx:
            mx = mn + 1.0
        
        universe = np.linspace(mn, mx, 100)
        var = fis.add_antecedent(f, universe)
        
        mid = (mn + mx) / 2
        var.add_term(term_labels[0], [mn, mn, mid])
        var.add_term(term_labels[1], [mn, mid, mx])
        var.add_term(term_labels[2], [mid, mx, mx])
        
        antecedents_dict[f] = var
    
    # === Консеквент ===
    y_mn, y_mx = float(y_train.min()), float(y_train.max())
    if y_mn == y_mx:
        y_mx = y_mn + 1.0
    
    y_universe = np.linspace(y_mn, y_mx, 100)
    consequent = fis.add_consequent('y_pred', y_universe)
    
    y_mid = (y_mn + y_mx) / 2
    consequent.add_term(term_labels[0], [y_mn, y_mn, y_mid])
    consequent.add_term(term_labels[1], [y_mn, y_mid, y_mx])
    consequent.add_term(term_labels[2], [y_mid, y_mx, y_mx])
    
    # === Генерация правил на основе данных ===
    # Квантильные границы
    bins = {}
    for f in feats:
        q33 = X_sub[f].quantile(0.33)
        q66 = X_sub[f].quantile(0.66)
        bins[f] = [q33, q66]
    
    y_q33 = y_train.quantile(0.33)
    y_q66 = y_train.quantile(0.66)
    
    # Полный перебор комбинаций термов
    for combo in itertools.product(term_labels, repeat=len(feats)):
        # Находим записи, попадающие в эту комбинацию
        mask = pd.Series(True, index=X_sub.index)
        for f, term in zip(feats, combo):
            if term == term_labels[0]:  # низкий
                mask &= (X_sub[f] <= bins[f][0])
            elif term == term_labels[1]:  # средний
                mask &= (X_sub[f] > bins[f][0]) & (X_sub[f] <= bins[f][1])
            else:  # высокий
                mask &= (X_sub[f] > bins[f][1])
        
        if mask.sum() > 0:
            mean_y = y_train[mask].mean()
            # Определяем выходной терм
            if mean_y <= y_q33:
                out_term = term_labels[0]
            elif mean_y <= y_q66:
                out_term = term_labels[1]
            else:
                out_term = term_labels[2]
            
            # Создаём правило
            conditions = {f: term for f, term in zip(feats, combo)}
            fis.add_rule(conditions, out_term)
    
    return {
        'fis': fis,
        'features': feats,
        'rules_count': len(fis.rules),
        'antecedents': antecedents_dict,
        'consequent': consequent,
        'term_labels': term_labels,
    }

# ==== Fuzzy logic type-2 ====

class IT2FuzzyVariable:
    def __init__(self, universe, name):
        self.universe = universe
        self.name = name
        self.terms_upper = {}
        self.terms_lower = {}

    def add_term(self, name, params_upper, params_lower):
        """params = [a, b, c] для треугольной функции"""
        self.terms_upper[name] = trimf(self.universe, params_upper)
        self.terms_lower[name] = trimf(self.universe, params_lower)

class IT2MamdaniFIS:
    def __init__(self):
        self.antecedents = {}
        self.consequent = None
        self.rules = []

    def add_antecedent(self, name, universe):
        var = IT2FuzzyVariable(universe, name)
        self.antecedents[name] = var
        return var

    def add_consequent(self, name, universe):
        self.consequent = IT2FuzzyVariable(universe, name)
        return self.consequent

    def add_rule(self, conditions, output_term):
        self.rules.append({'conditions': conditions, 'output_term': output_term})

    def compute(self, inputs):
        if self.consequent is None: return np.nan
        
        # Агрегация верхних и нижних функций принадлежности выхода
        aggregated_upper = np.zeros_like(self.consequent.universe)
        aggregated_lower = np.zeros_like(self.consequent.universe)
        
        for rule in self.rules:
            firing_upper = 1.0
            firing_lower = 1.0
            for var_name, term_name in rule['conditions'].items():
                if var_name not in inputs: return np.nan
                val = inputs[var_name]
                mu_up = float(np.interp(val, self.antecedents[var_name].universe, self.antecedents[var_name].terms_upper[term_name]))
                mu_low = float(np.interp(val, self.antecedents[var_name].universe, self.antecedents[var_name].terms_lower[term_name]))
                firing_upper = min(firing_upper, mu_up)
                firing_lower = min(firing_lower, mu_low)
            
            if firing_upper > 0:
                out_term_up = self.consequent.terms_upper[rule['output_term']]
                out_term_low = self.consequent.terms_lower[rule['output_term']]
                aggregated_upper = np.maximum(aggregated_upper, np.minimum(out_term_up, firing_upper))
                aggregated_lower = np.maximum(aggregated_lower, np.minimum(out_term_low, firing_lower))
                
        # Редукция типа Nie-Tan (усреднение верхней и нижней границ)
        total = np.sum(aggregated_upper + aggregated_lower)
        if total == 0: return float(np.mean(self.consequent.universe))
        return float(np.sum(self.consequent.universe * (aggregated_upper + aggregated_lower)) / total)

def build_fuzzy_type2_from_data(X_train, y_train, feature_names, max_features=3):
    import pandas as pd
    feats = feature_names[:max_features]
    X_sub = X_train[feats].copy()
    
    fis = IT2MamdaniFIS()
    term_labels = ['низкий', 'средний', 'высокий']
    antecedents_dict = {}
    
    for f in feats:
        mn, mx = float(X_sub[f].min()), float(X_sub[f].max())
        if mn == mx: mx = mn + 1.0
        universe = np.linspace(mn, mx, 100)
        var = fis.add_antecedent(f, universe)
        
        mid = (mn + mx) / 2
        # Верхняя ФП (широкая), Нижняя ФП (узкая) - Footprint of Uncertainty
        var.add_term(term_labels[0], [mn, mn, mid], [mn+0.1*(mid-mn), mn+0.1*(mid-mn), mid-0.1*(mid-mn)])
        var.add_term(term_labels[1], [mn, mid, mx], [mn+0.1*(mid-mn), mid, mx-0.1*(mid-mn)])
        var.add_term(term_labels[2], [mid, mx, mx], [mid+0.1*(mx-mid), mx-0.1*(mx-mid), mx])
        antecedents_dict[f] = var

    y_mn, y_mx = float(y_train.min()), float(y_train.max())
    if y_mn == y_mx: y_mx = y_mn + 1.0
    y_universe = np.linspace(y_mn, y_mx, 100)
    consequent = fis.add_consequent('y_pred', y_universe)
    y_mid = (y_mn + y_mx) / 2
    consequent.add_term(term_labels[0], [y_mn, y_mn, y_mid], [y_mn+0.1*(y_mid-y_mn), y_mn+0.1*(y_mid-y_mn), y_mid-0.1*(y_mid-y_mn)])
    consequent.add_term(term_labels[1], [y_mn, y_mid, y_mx], [y_mn+0.1*(y_mid-y_mn), y_mid, y_mx-0.1*(y_mx-y_mid)])
    consequent.add_term(term_labels[2], [y_mid, y_mx, y_mx], [y_mid+0.1*(y_mx-y_mid), y_mx-0.1*(y_mx-y_mid), y_mx])

    # Генерация правил (аналогично Type-1)
    bins = {f: [X_sub[f].quantile(0.33), X_sub[f].quantile(0.66)] for f in feats}
    y_q33, y_q66 = y_train.quantile(0.33), y_train.quantile(0.66)
    
    for combo in itertools.product(term_labels, repeat=len(feats)):
        mask = pd.Series(True, index=X_sub.index)
        for f, term in zip(feats, combo):
            if term == term_labels[0]: mask &= (X_sub[f] <= bins[f][0])
            elif term == term_labels[1]: mask &= (X_sub[f] > bins[f][0]) & (X_sub[f] <= bins[f][1])
            else: mask &= (X_sub[f] > bins[f][1])
            
        if mask.sum() > 0:
            mean_y = y_train[mask].mean()
            out_term = term_labels[0] if mean_y <= y_q33 else (term_labels[1] if mean_y <= y_q66 else term_labels[2])
            fis.add_rule({f: term for f, term in zip(feats, combo)}, out_term)
            
    return {'fis': fis, 'features': feats, 'antecedents': antecedents_dict, 'consequent': consequent, 'term_labels': term_labels}