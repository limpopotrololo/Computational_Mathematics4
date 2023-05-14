import math


def line_func(x, a, b):
    return a * x + b


def aprox(x, y, n):
    a = (n * sum([xi * yi for xi, yi in zip(x, y)]) - sum(x) * sum(y)) / (
            n * sum([xi ** 2 for xi in x]) - (sum(x) ** 2))
    b = (sum(y) - a * sum(x)) / n

    sigma = 0
    eps = []
    for i in range(n):
        eps.append(line_func(x[i], a, b) - y[i])
        sigma += (line_func(x[i], a, b) - y[i]) ** 2
    delta = math.sqrt(sigma / n)
    y_aprox = []
    for i in range(n):
        y_aprox.append(line_func(x[i], a, b))

    xy = sum([xi * yi for xi, yi in zip(x, y_aprox)])
    x_mean = sum(x) / n
    y_mean = sum(y_aprox) / n
    x_std = math.sqrt(sum([(xi - x_mean) ** 2 for xi in x]) / n)
    y_std = math.sqrt(sum([(yi - y_mean) ** 2 for yi in y_aprox]) / n)
    r_pirson = (xy - n * x_mean * y_mean) / (n * x_std * y_std)
    function = f"{round(a, 3)} * x{round(b, 3)})"
    return x, y, y_aprox, eps, sigma, delta, r_pirson, a, b, function
