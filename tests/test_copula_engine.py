import numpy as np

from copula_engine import pseudo_obs, CopulaAnalyzer, GaussianCopulaRegressor


def test_pseudo_obs_ranks():
    data = np.array([10, 20, 30])
    res = pseudo_obs(data)
    assert len(res) == 3
    assert abs(res[0] - 1/4) < 1e-8
    assert abs(res[1] - 2/4) < 1e-8
    assert abs(res[2] - 3/4) < 1e-8


def test_copula_analyzer_fit_all():
    np.random.seed(0)
    rho = 0.7
    n = 500
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal([0, 0], cov, size=n)
    x = data[:, 0]
    y = data[:, 1]

    ca = CopulaAnalyzer(x, y)
    results = ca.fit_all()

    # Check expected copula keys present
    for key in ["Gaussian", "Clayton", "Gumbel", "Frank", "Student-t"]:
        assert key in results

    # Gaussian estimate should be close to true rho
    est_rho = results['Gaussian']['param']
    assert abs(est_rho - rho) < 0.15


def test_gaussian_copula_regressor_predicts():
    np.random.seed(1)
    n = 200
    X = np.random.randn(n, 2)
    y = 2 * X[:, 0] - 1.0 * X[:, 1] + np.random.randn(n) * 0.1

    reg = GaussianCopulaRegressor()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    y_pred = np.asarray(y_pred).ravel()
    assert y_pred.shape[0] == n

    # Predictions should correlate reasonably with true y
    corr = np.corrcoef(y, y_pred)[0, 1]
    assert corr > 0.5
