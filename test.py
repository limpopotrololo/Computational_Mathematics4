# import numpy as np
#
#
# def log_aprox(x, y):
#     """
#     Аппроксимация логарифмической функцией y = a*log(x) + b
#     :param x: numpy.array - массив значений аргумента x
#     :param y: numpy.array - массив значений функции y
#     :return: tuple(a, b, y_aprox) - кортеж, содержащий коэффициенты a, b
#                                    и массив аппроксимированных значений y
#     """
#     n = len(x)
#     if n != len(y):
#         raise ValueError("Длины массивов x и y не совпадают")
#     if any(x <= 0):
#         raise ValueError("Аргумент x должен быть больше нуля")
#
#     log_x = np.log(x)
#     log_y = np.log(y)
#
#     sum_log_x = np.sum(log_x)
#     sum_log_y = np.sum(log_y)
#     sum_log_x2 = np.sum(log_x ** 2)
#     sum_log_xy = np.sum(log_x * log_y)
#
#     delta = n * sum_log_x2 - sum_log_x ** 2
#     a = (n * sum_log_xy - sum_log_x * sum_log_y) / delta
#     b = (sum_log_x2 * sum_log_y - sum_log_x * sum_log_xy) / delta
#
#     y_aprox = np.exp(b) * x ** a
#
#     return a, b, y_aprox
# import matplotlib.pyplot as plt
#
# # Данные для аппроксимации
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# y = np.array([2.5, 4, 6.1, 7.8, 9.7, 11.5, 13.2, 15.1])
#
# # Аппроксимация
# a, b, y_aprox = log_aprox(x, y)
#
# # Вывод результатов
# print(f"a = {a}")
# print(f"b = {b}")

# График
# plt.plot(x, y, 'ro', label='Исходные данные')
# plt.plot(x, y_aprox, label='Аппроксимация')
# plt.legend()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
#
#
# def exp_f(x, a, b):
#     return a * np.exp(b * x)
#
# # Данные для аппроксимации
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# y = np.array([2.5,4,6.1,7.8,9.7,11.5,13.2,15.1])
#
# popt, pcov = curve_fit(exp_f, x, y)
#
# # Коэффициенты аппроксимации
# a = popt[0]
# b = popt[1]
#
# # Вывод результатов
# print(f"a = {a}")
# print(f"b = {b}")
#
# # График
# plt.plot(x, y, 'o', label='исходные данные')
# plt.plot(x, exp_f(x, *popt), '-', label='аппроксимация')
# plt.legend()
# plt.show()
#
# def start():
#     return 1,1
# a,b = start()
# a,b = start()
# a,b =  start()
# print(a)
# print(b)
# a,b = input("Введите числа").split(" ")
# a = float(a)
# b = float(b)
# print(f"{round(a, 3)} * e^({round(b, 3)}x)")
# def f(x):
#     return x ** 3 + 4.81 * x ** 2-17.37 * x + 5.38
#
# def chord_method(f, a, b, tol, max_iter=100):
#     """
#     Solves a nonlinear equation f(x) = 0 using the chord method on the interval [a, b].
#     Returns a table with columns: iteration count, a, b, x_k, f(a), f(b), f(x), |x_k - x_{k-1}|.
#     Stops when either the absolute error is smaller than tol or the maximum number of iterations is reached.
#     """
#     # Initialize the table
#     table = [['Iteration', 'a', 'b', 'x_k', 'f(a)', 'f(b)', 'f(x_k)', '|x_k - x_{k-1}|']]
#     x_k_prev = a
#     x_k = b
#     fa = f(a)
#     fb = f(b)
#     fxk_prev = f(x_k_prev)
#     fxk = f(x_k)
#     iter_count = 0
#     # Iteration loop
#     while abs(x_k - x_k_prev) > tol and iter_count < max_iter:
#         iter_count += 1
#         # Compute x_k+1 using the chord formula
#         x_k_next = x_k - fxk*(x_k - x_k_prev)/(fxk - fxk_prev)
#         fxk_next = f(x_k_next)
#         # Update the table and variables
#         row = [iter_count, a, b, x_k_next, fa, fb, fxk_next, abs(x_k_next - x_k_prev)]
#         table.append(row)
#         x_k_prev = x_k
#         x_k = x_k_next
#         fxk_prev = fxk
#         fxk = fxk_next
#         # Update fa and fb to keep track of the signs of f(a) and f(b)
#         if fa*fxk < 0:
#             fb = fxk
#             b = x_k
#         else:
#             fa = fxk
#             a = x_k
#     return table
#
#
# table = chord_method(f, -8, -7, tol=0.001)
# for row in table:
#     print(row)


# def newton_method(f, df, x_0, tol=1e-6, max_iter=100):
#     """
#     Solves a nonlinear equation f(x) = 0 using the Newton's method starting from x_0.
#     Returns a table with columns: iteration count, x_k, f(x_k), f'(x_k), x_{k+1}, |x_{k+1} - x_k|.
#     Stops when either the absolute error is smaller than tol or the maximum number of iterations is reached.
#     """
#     # Initialize the table
#     table = [['Iteration', 'x_k', 'f(x_k)', 'f\'(x_k)', 'x_{k+1}', '|x_{k+1} - x_k|']]
#     x_k = x_0
#     fxk = f(x_k)
#     dfxk = df(x_k)
#     iter_count = 0
#     # Iteration loop
#     while abs(fxk/dfxk) > tol and iter_count < max_iter:
#         iter_count += 1
#         # Compute x_k+1 using the Newton's formula
#         x_k_next = x_k - fxk/dfxk
#         fxk_next = f(x_k_next)
#         dfxk_next = df(x_k_next)
#         # Update the table and variables
#         row = [iter_count, x_k, fxk, dfxk, x_k_next, abs(x_k_next - x_k)]
#         table.append(row)
#         x_k = x_k_next
#         fxk = fxk_next
#         dfxk = dfxk_next
#     return table
# def f(x):
#     return x ** 3 + 4.81 * x ** 2-17.37 * x + 5.38
#
# def df(x):
#     return 3 * x ** 2 + 9.62 * x - 17.37
#
# table = newton_method(f, df, 1)
# for row in table:
#     print(row)
import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return (31*x)/(x**4 + 13)
x = 0
yl = []
xl = []

for i in range(11):
    xl.append(round(x,4))
    yl.append(round(f(x),4))
    print(x,f(x))
    x+=0.4
print(xl)
print(yl)
x = xl
y = yl
n = len(x)

# вычисление коэффициентов a и b
a = (n*np.sum([x[i]*y[i] for i in range(n)]) - np.sum(x)*np.sum(y)) / (n*np.sum([x[i]**2 for i in range(n)]) - np.sum(x)**2)
b = (np.sum(y) - a*np.sum(x)) / n

# уравнение прямой
line_eq = f"y = {a:.2f}x + {b:.2f}"
print(line_eq)
y_approx = [a*x[i] + b for i in range(n)]

# среднеквадратическое отклонение
mse = np.sqrt(np.sum([(yl[i] - y_approx[i])**2 for i in range(n)]) / n)
print(f"Среднеквадратичное отклонение: {mse:.4f}")
# x = np.array([0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4])
# y = np.array([0, 0.9519715022724481, 1.8494213101061927, 2.467890882071967,
#               2.5366172980934456, 2.1379310344827585, 1.6111707841031149,
#               1.1656388990352593, 0.8416937049456295, 0.6167054225868915,
#               0.4609665427509295])
#
# n = len(x)
#
# # Составляем систему уравнений
# A = np.array([[np.sum(x**4), np.sum(x**3), np.sum(x**2)],
#               [np.sum(x**3), np.sum(x**2), np.sum(x)],
#               [np.sum(x**2), np.sum(x), n]])
# B = np.array([np.sum(x**2 * y), np.sum(x * y), np.sum(y)])
# coeffs = np.linalg.solve(A, B)
#
# # Квадратное уравнение
# a, b, c = coeffs
# quad_func = lambda x: a*x**2 + b*x + c
# print(f"{round(a,2)}x^2+{round(b,2)}x+{round(c,2)}")
# # Создаем массив x_vals для построения графика
# quad_vals = quad_func(x)
#
# # Построение графиков
# plt.scatter(x, y, label='Заданная функция')
# plt.plot(x, quad_vals, label='Квадратичная аппроксимация')
# plt.legend()
# plt.show()
# # Находим среднеквадратичное отклонение
# mse = np.sum((quad_vals - y)**2) / n
# print('Среднеквадратичное отклонение для квадратичной аппроксимации:', round(mse, 3))

# Оформление
# построение графика
# plt.scatter(x, y, label='Исходные точки')
# plt.plot(x, [a*x[i]+b for i in range(n)], label=line_eq)
# plt.legend()
# plt.show()
# a = []
# sum = 0
# sum += (f(3 / 4))
# sum += (f(5 / 4))
# sum += (f(7 / 4))
# sum += (f(9 / 4))
# sum += (f(11 / 4))
# print(sum * (2 / 5))
# x=1
# sum = 0
# for i in range(10):
#     sum+=f(x)
#     print(f(x),x)
#     x += 0.2
# print(f(3),3)
# print((2*(sum-f(1))+f(3)+f(1))*2/20)
# def simpson_rule(f, a, b, n):
#     """
#     Вычисление интеграла f на отрезке [a, b] методом Симпсона
#     с использованием n интервалов.
#     """
#     h = (b - a) / n  # длина интервала
#     x = [a + i * h for i in range(n + 1)]  # точки разбиения
#     fx = [f(x[i]) for i in range(n + 1)]  # значения функции в точках разбиения
#
#     # вычисление значения интеграла методом Симпсона
#     sum_odd = sum(fx[i] for i in range(1, n, 2))
#     sum_even = sum(fx[i] for i in range(2, n, 2))
#     integral = h / 3 * (fx[0] + 4 * sum_odd + 2 * sum_even + fx[n])
#
#     return integral
# print(simpson_rule(f,1,3,10))
