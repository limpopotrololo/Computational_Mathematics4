import numpy as np
from linear_aprox import aprox as ln
import math


def exp_f(x, a, b):
    return a * np.exp(b * x)


def aprox(x, y, n):
    if not all(xi > 0 for xi in x):
        print("Значение x должно быть больше 0")
        return None
    y_line = np.log(y)
    r_x, r_y, r_y_aprox, r_eps, r_sigma, r_delta, r_r_pirson, b, a, formula = ln(x, y_line, n)
    a = np.exp(a)
    eps = []
    sigma = 0
    y_aprox = []
    for i in range(n):
        y_aprox.append(exp_f(x[i], a, b))
        eps.append(exp_f(x[i], a, b) - y[i])
        sigma += (exp_f(x[i], a, b) - y[i]) ** 2
    delta = math.sqrt(sigma / n)
    # xy = sum([xi * yi for xi, yi in zip(x, y_aprox)])
    # x_mean = sum(x) / n
    # y_mean = sum(y_aprox) / n
    # x_std = math.sqrt(sum([(xi - x_mean) ** 2 for xi in x]) / n)
    # y_std = math.sqrt(sum([(yi - y_mean) ** 2 for yi in y_aprox]) / n)
    r_pirson = None
    function = f"{round(a, 3)} * e^({round(b, 3)}x)"
    return x, y, y_aprox, eps, sigma, delta, r_pirson, a, b, function
