import numpy as np
from linear_aprox import aprox as ln
import math
from sklearn.linear_model import LinearRegression


def power_f(x, a, b):
    return a * x ** b


def power(x, y, n):
    if not all(xi > 0 for xi in x):
        print("Значение x должно быть больше 0")
        return None

    log_x = np.log(x)
    log_y = np.log(y)
    r_x, r_y, r_y_aprox, r_eps, r_sigma, r_delta, r_r_pirson, b, a, formula = ln(log_x, log_y, n)
    a = np.exp(a)
    y_approx = []
    for i in range(n):
        y_approx.append(power_f(x[i], a, b))
    eps = []
    sigma = 0
    for i in range(n):
        eps.append(y_approx[i] - y[i])
        sigma += (y_approx[i] - y[i]) ** 2
    delta = math.sqrt(sigma / n)
    r_pirson = None
    function = f"{round(a, 3)} * x^{round(b, 3)}"

    return x, y, y_approx, eps, sigma, delta, r_pirson, a, b, function
