# copula_engine.py
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def pseudo_obs(data):
    """Преобразование данных в псевдонаблюдения (ранги / (n+1))"""
    n = len(data)
    return stats.rankdata(data) / (n + 1)

class CopulaAnalyzer:
    def __init__(self, u, v):
        self.u = pseudo_obs(u)
        self.v = pseudo_obs(v)
        self.n = len(u)
        self.results = {}

    def _log_lik_gaussian(self, rho):
        rho = np.clip(rho[0], -0.99, 0.99)
        x = stats.norm.ppf(self.u)
        y = stats.norm.ppf(self.v)
        log_c = -0.5 * np.log(1 - rho**2) - (rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
        return -np.sum(log_c)

    def _log_lik_clayton(self, theta):
        theta = np.clip(theta[0], 0.01, 20)
        u, v = self.u, self.v
        log_c = np.log(1 + theta) - (1 + theta) * np.log(u**(-theta) + v**(-theta) - 1) - (1/theta + 2) * np.log(u * v)
        return -np.sum(log_c)

    def _log_lik_gumbel(self, theta):
        theta = np.clip(theta[0], 1.01, 20)
        u, v = self.u, self.v
        # Стабильная формула логарифма плотности копулы Гумбеля
        # c(u,v) = C(u,v) * (u*v)^(-1) * ((-ln u)*(-ln v))^(theta-1) * a^((1/theta)-2) * (a + (theta-1)) / theta
        # где a = (-ln u)^theta + (-ln v)^theta
        lu = -np.log(u)
        lv = -np.log(v)
        a = lu**theta + lv**theta
        C = np.exp(-a**(1/theta)) # Сама копула Gumbel

        # Логарифм плотности
        log_c = np.log(C) - np.log(u) - np.log(v) + (theta - 1) * (np.log(lu) + np.log(lv))
        log_c += (1/theta - 2) * np.log(a)
        log_c += np.log(a + (theta - 1)) - np.log(theta)

        return -np.sum(log_c) 

    def _log_lik_frank(self, theta):
        theta = np.clip(theta[0], -50, 50)
        if abs(theta) < 1e-6: return 1e10
        u, v = self.u, self.v
        num = (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
        den = (1 - np.exp(-theta*u) - np.exp(-theta*v) + np.exp(-theta*(u+v)))**2
        log_c = np.log(np.maximum(1e-10, num / den))
        return -np.sum(log_c)

    def _log_lik_student(self, params):
        rho, nu = params
        rho = np.clip(rho, -0.99, 0.99)
        nu = np.clip(nu, 2.1, 30)
        x = stats.t.ppf(self.u, nu)
        y = stats.t.ppf(self.v, nu)
        # Корректная логарифмическая функция правдоподобия для t-копулы
        # Плотность t-копулы: c(u,v) = t_2(x,y; Sigma, nu) / (t_1(x; nu) * t_1(y; nu))
        # log_c = log(t_2) - log(t_1(x)) - log(t_1(y))

        det_sigma = 1 - rho**2
        inv_sigma_quad = (x**2 - 2*rho*x*y + y**2) / det_sigma

        # Логарифм многомерной плотности t (с точностью до констант, которые сократятся)
        log_t2 = -0.5 * np.log(det_sigma) - ((nu + 2) / 2) * np.log(1 + inv_sigma_quad / nu)

        # Логарифм маргинальных плотностей
        log_t1_x = -((nu + 1) / 2) * np.log(1 + x**2 / nu)
        log_t1_y = -((nu + 1) / 2) * np.log(1 + y**2 / nu)

        log_c = log_t2 - log_t1_x - log_t1_y
        return -np.sum(log_c)

    def fit_all(self):
        # Gaussian
        res = minimize(self._log_lik_gaussian, [0.5], method='L-BFGS-B', bounds=[(-0.99, 0.99)])
        rho = res.x[0]
        self.results['Gaussian'] = {'param': rho, 'loglik': -res.fun, 'aic': 2*1 + 2*res.fun, 'bic': np.log(self.n)*1 + 2*res.fun, 'tail_l': 0, 'tail_u': 0}

        # Clayton
        res = minimize(self._log_lik_clayton, [1.0], method='L-BFGS-B', bounds=[(0.01, 20)])
        theta = res.x[0]
        self.results['Clayton'] = {'param': theta, 'loglik': -res.fun, 'aic': 2*1 + 2*res.fun, 'bic': np.log(self.n)*1 + 2*res.fun, 'tail_l': 2**(-1/theta), 'tail_u': 0}

        # Gumbel
        res = minimize(self._log_lik_gumbel, [1.5], method='L-BFGS-B', bounds=[(1.01, 20)])
        theta = res.x[0]
        self.results['Gumbel'] = {'param': theta, 'loglik': -res.fun, 'aic': 2*1 + 2*res.fun, 'bic': np.log(self.n)*1 + 2*res.fun, 'tail_l': 0, 'tail_u': 2 - 2**(1/theta)}

        # Frank
        res = minimize(self._log_lik_frank, [1.0], method='L-BFGS-B', bounds=[(-50, 50)])
        theta = res.x[0]
        self.results['Frank'] = {'param': theta, 'loglik': -res.fun, 'aic': 2*1 + 2*res.fun, 'bic': np.log(self.n)*1 + 2*res.fun, 'tail_l': 0, 'tail_u': 0}

        # Student-t (упрощенно инициализируем nu=5)
        res = minimize(self._log_lik_student, [0.5, 5], method='L-BFGS-B', bounds=[(-0.99, 0.99), (2.1, 30)])
        rho, nu = res.x
        # Формула хвостовой зависимости для Стьюдента
        tail = 2 * stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), nu+1)
        self.results['Student-t'] = {'param': (rho, nu), 'loglik': -res.fun, 'aic': 2*2 + 2*res.fun, 'bic': np.log(self.n)*2 + 2*res.fun, 'tail_l': tail, 'tail_u': tail}
        
        return self.results

class GaussianCopulaRegressor:
    """Регрессор на основе Гауссовой копулы"""
    def fit(self, X, y):
        self.X_means = X.mean(axis=0)
        self.X_stds = X.std(axis=0)
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # Преобразование в нормальные оценки
        X_norm = np.array([stats.norm.ppf(stats.rankdata(x) / (len(x)+1)) for x in X.T]).T
        y_norm = stats.norm.ppf(stats.rankdata(y) / (len(y)+1))
        
        # Линейная регрессия в латентном пространстве
        self.model = np.linalg.lstsq(np.column_stack([np.ones(len(X_norm)), X_norm]), y_norm, rcond=None)[0]
        
    def predict(self, X):
        X_norm = np.array([stats.norm.ppf(stats.rankdata(x) / (len(x)+1)) for x in X.T]).T
        # Если X - одна строка
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(1, -1)
        y_pred_norm = np.column_stack([np.ones(len(X_norm)), X_norm]) @ self.model
        # Обратное преобразование
        y_pred = stats.norm.cdf(y_pred_norm) * (self.y_std * 3) + (self.y_mean - 1.5 * self.y_std) # Грубая калибровка для диапазона
        # Лучше использовать квантили исходного y
        sorted_y = np.sort(np.array([self.y_mean - 1.5*self.y_std, self.y_mean + 1.5*self.y_std])) # Заглушка, лучше использовать empirical distribution
        return np.interp(stats.norm.cdf(y_pred_norm), [0, 1], sorted_y)