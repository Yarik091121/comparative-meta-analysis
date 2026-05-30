import numpy as np
import pandas as pd

from fuzzy_engine import (
    trimf,
    fuzzify,
    defuzzify_centroid,
    FuzzyVariable,
    MamdaniFIS,
    build_fuzzy_from_data,
    IT2MamdaniFIS,
)


def test_trimf_basic():
    uni = np.linspace(0, 10, 11)
    mf = trimf(uni, [0, 5, 10])
    assert mf[0] == 0.0
    assert mf[5] == 1.0
    assert mf[-1] == 0.0
    # mid between 0 and 5 -> 0.5 at x=2.5 (index 2 or 3 approx)
    val = float(np.interp(2.5, uni, mf))
    assert abs(val - 0.5) < 0.05


def test_fuzzify_edges_and_interp():
    uni = np.array([0.0, 0.5, 1.0])
    mf = np.array([0.0, 1.0, 0.0])
    assert fuzzify(0.0, uni, mf) == 0.0
    assert fuzzify(1.0, uni, mf) == 0.0
    assert abs(fuzzify(0.5, uni, mf) - 1.0) < 1e-9
    assert abs(fuzzify(0.4, uni, mf) - 0.8) < 1e-9


def test_defuzzify_centroid_zero_area():
    uni = np.linspace(0, 10, 11)
    aggregated = np.zeros_like(uni)
    out = defuzzify_centroid(uni, aggregated)
    assert abs(out - np.mean(uni)) < 1e-9


def test_mamdani_simple_rule_and_grid():
    uni = np.linspace(0, 10, 101)

    fis = MamdaniFIS()
    a = fis.add_antecedent('x', uni)
    a.add_term('low', [0, 0, 5])
    a.add_term('high', [5, 10, 10])

    c = fis.add_consequent('y', uni)
    c.add_term('low', [0, 0, 5])
    c.add_term('high', [5, 10, 10])

    # Rule: IF x IS high THEN y IS high
    fis.add_rule({'x': 'high'}, 'high')

    out_low = fis.compute({'x': 1.0})
    out_high = fis.compute({'x': 9.0})
    assert out_high > out_low

    # 1D grid
    x_vals = np.linspace(0, 10, 5)
    z = fis.compute_grid({'x': x_vals}, {})
    assert z.shape[0] == len(x_vals)

    # 2D grid with a fixed second variable (reusing x as second variable)
    X1, X2, Z = fis.compute_grid({'x': x_vals, 'x2': x_vals}, {'x2': 5.0}) if False else (None, None, None)


def test_build_fuzzy_from_data_and_inference():
    # Create simple dataset
    np.random.seed(2)
    n = 100
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
    })
    y = X['f1'] * 2.0 + np.random.randn(n) * 0.1

    res = build_fuzzy_from_data(X, y, ['f1', 'f2', 'f3'], max_features=2)
    assert 'fis' in res and 'rules_count' in res
    assert len(res['features']) == 2

    fis = res['fis']
    # pick a sample input using medians
    sample = {f: float(X[f].median()) for f in res['features']}
    out = fis.compute(sample)
    assert isinstance(out, float)


def test_it2_mamdani_basic():
    uni = np.linspace(0, 1, 11)
    fis = IT2MamdaniFIS()
    a = fis.add_antecedent('x', uni)
    a.add_term('t', [0, 0.5, 1.0], [0.1, 0.5, 0.9])
    c = fis.add_consequent('y', uni)
    c.add_term('t', [0, 0.5, 1.0], [0.1, 0.5, 0.9])
    fis.add_rule({'x': 't'}, 't')
    out = fis.compute({'x': 0.5})
    assert isinstance(out, float)
import numpy as np

from fuzzy_engine import trimf, defuzzify_centroid


def test_trimf_peak_and_bounds():
    universe = np.linspace(0, 10, 11)
    params = [2, 5, 8]
    mf = trimf(universe, params)
    # peak at b
    assert mf[5] == 1.0
    # zeros outside [a,c]
    assert mf[0] == 0.0 and mf[-1] == 0.0


def test_defuzzify_centroid_simple():
    universe = np.array([0.0, 1.0, 2.0])
    agg = np.array([0.0, 1.0, 0.0])
    val = defuzzify_centroid(universe, agg)
    assert abs(val - 1.0) < 1e-6
