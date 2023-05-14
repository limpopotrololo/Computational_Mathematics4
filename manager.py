from linear_aprox import aprox as ln
from log_aprox import aprox as log
from exp_aprox import aprox as exp
from polyn_aprox import aprox as pol
from drawer import *
from output_module import *

def action(x, y, n, flag):
    fx = []
    fy = []
    y_aprox = []
    eps = []
    sigma = []
    delta = []
    r_pirson = []
    functions = [ln, log, exp, (pol, 2), (pol, 3)]
    formula = []
    name = ["Линейная", "Логарифмическая", "Экспоненциальная", "Полиномиальная 2ой степени",
            "Полиномиальная 3ей степени"]

    for func in functions:
        if isinstance(func, tuple):
            func_name, degree = func
            # Если функция принимает несколько аргументов, передаем их в функцию через распаковку кортежа
            rx, ry, r_y_aprox, r_eps, r_sigma, r_delta, r_r_pirson, r_formula = func_name(x, y, n, degree)
        else:
            rx, ry, r_y_aprox, r_eps, r_sigma, r_delta, r_r_pirson, a, b, r_formula = func(x, y, n)
        fx.append(rx)
        fy.append(ry)
        y_aprox.append(r_y_aprox)
        eps.append(r_eps)
        sigma.append(r_sigma)
        delta.append(r_delta)
        r_pirson.append(r_r_pirson)
        formula.append(r_formula)
    output = "\nВЫВОД:\n"
    for i in range(5):
        output += name[i] + " аппроксимирующая функция:\n"
        output += "Исходный ряд X: \n" + " ".join(map(str, x)) + "\n"
        output += "Исходный ряд Y: \n" + " ".join(map(str, y)) + "\n"
        output += "Значения аппроксимирующей функции: \n" + " ".join(
            map(str, [round(val, 4) for val in y_aprox[i]])) + "\n"
        output += "Отклонения: \n" + " ".join(map(str, [round(val, 4) for val in eps[i]])) + "\n"
        output += "Квадратичное отклонение = " + str(round(delta[i], 4)) + "\n"
        if name[i] == "Линейная":
            output += "Коэффициент Пирсона = " + str(round(r_pirson[i], 4)) + "\n"
        output += "Итоговая формула " + str(formula[i]) + "\n"
        plot_approximation(x, y, y_aprox[i], str(formula[i]))
        output += "--------------------------------------------------------------\n"
    max_delta = min(delta)
    best_index = delta.index(max_delta)
    output += "Лучшая аппроксимирующая функция - " + str(name[best_index]) + "\n"
    output += "Ее вид: " + str(formula[best_index])+"\n"
    if flag == "console":
        console_output(output)
    else:
        file_output(output)
