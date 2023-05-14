import numpy as np


def polyn_2(x, a, b, c):
    return a * x ** 2 + b * x + c


def polyn_3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def aprox(x, y, n, k):
    if (k == 2):
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum([xi ** 2 for xi in x])
        sum_x3 = sum([xi ** 3 for xi in x])
        sum_x4 = sum([xi ** 4 for xi in x])
        sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
        sum_x2y = sum([xi ** 2 * yi for xi, yi in zip(x, y)])

        A = np.array([[sum_x4, sum_x3, sum_x2],
                      [sum_x3, sum_x2, sum_x],
                      [sum_x2, sum_x, n]])

        B = np.array([sum_x2y, sum_xy, sum_y])

        a, b, c = np.linalg.solve(A, B)

        y_approx = [polyn_2(xi, a, b, c) for xi in x]
        eps = [y[i] - y_approx[i] for i in range(n)]

        sigma2 = sum([epsi ** 2 for epsi in eps])

        delta = np.sqrt(sigma2 / n)

        xy = sum([xi * y_approx[i] for i, xi in enumerate(x)])
        x_mean = sum(x) / n
        y_mean = sum(y_approx) / n
        x_std = np.sqrt(sum([(xi - x_mean) ** 2 for xi in x]) / n)
        y_std = np.sqrt(sum([(yi - y_mean) ** 2 for yi in y]) / n)

        r_pirson = (xy - n * x_mean * y_mean) / (n * x_std * y_std)
        function =  f"{round(a, 3)}x^2 + ({round(b, 3)})x + ({round(c,3)}))"
        return x, y, y_approx, eps, sigma2, delta, r_pirson, function

    elif k == 3:
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum([xi ** 2 for xi in x])
        sum_x3 = sum([xi ** 3 for xi in x])
        sum_x4 = sum([xi ** 4 for xi in x])
        sum_x5 = sum([xi ** 5 for xi in x])
        sum_x6 = sum([xi ** 6 for xi in x])
        sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
        sum_x2y = sum([xi ** 2 * yi for xi, yi in zip(x, y)])
        sum_x3y = sum([xi ** 3 * yi for xi, yi in zip(x, y)])

        A = np.array([[sum_x6, sum_x5, sum_x4, sum_x3],
                      [sum_x5, sum_x4, sum_x3, sum_x2],
                      [sum_x4, sum_x3, sum_x2, sum_x],
                      [sum_x3, sum_x2, sum_x, n]])

        B = np.array([sum_x3y, sum_x2y, sum_xy, sum_y])

        a, b, c, d = np.linalg.solve(A, B)

        y_approx = [polyn_3(xi, a, b, c, d) for xi in x]

        eps = [y[i] - y_approx[i] for i in range(n)]

        sigma3 = sum([epsi ** 2 for epsi in eps])

        delta = np.sqrt(sigma3 / n)

        # xy = sum([xi * y_approx[i] for i, xi in enumerate(x)])
        # x_mean = sum(x) / n
        # y_mean = sum(y_approx) / n
        # x_std = np.sqrt(sum([(xi - x_mean) ** 2 for xi in x]) / n)
        # y_std = np.sqrt(sum([(yi - y_mean) ** 2 for yi in y]) / n)

        r_pirson = None
        function = f"{round(a, 3)}x^3 + ({round(b, 3)})x^2 + ({round(c,3)})x + ({round(d,3)})"
        return x, y, y_approx, eps, sigma3, delta, r_pirson, function
    else:
        return None
