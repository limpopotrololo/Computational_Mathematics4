import numpy as np
from linear_aprox import aprox as ln
import math


def log_f(x, a, b):
    return a * np.log(x) + b


def aprox(x, y, n):
    if not all(xi > 0 for xi in x):
        print("Значение x должно быть больше 0")
        return None
    log_x = np.log(x)
    log_y = np.log(y)

    sum_log_x = np.sum(log_x)
    sum_log_y = np.sum(log_y)
    sum_log_x2 = np.sum(log_x ** 2)
    sum_log_xy = np.sum(log_x * log_y)

    delta = n * sum_log_x2 - sum_log_x ** 2
    a = (n * sum_log_xy - sum_log_x * sum_log_y) / delta
    b = (sum_log_x2 * sum_log_y - sum_log_x * sum_log_xy) / delta

    y_aprox = np.exp(b) * x ** a
    eps = []
    sigma = 0
    for i in range(n):
        eps.append(y_aprox[i] - y[i])
        sigma += (y_aprox[i] - y[i]) ** 2
    delta = math.sqrt(sigma / n)
    # xy = sum([xi * yi for xi, yi in zip(x, y_aprox)])
    # x_mean = sum(x) / n
    # y_mean = sum(y_aprox) / n
    # x_std = math.sqrt(sum([(xi - x_mean) ** 2 for xi in x]) / n)
    # y_std = math.sqrt(sum([(yi - y_mean) ** 2 for yi in y_aprox]) / n)
    r_pirson = None
    function = f"{round(a, 3)} * log(x) + {round(b, 3)})"
    return x, y, y_aprox, eps, sigma, delta, r_pirson, a, b, function
