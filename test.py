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
a,b = input("Введите числа").split(" ")
a = float(a)
b = float(b)
print(f"{round(a, 3)} * e^({round(b, 3)}x)")