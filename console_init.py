from manager import action


def console_init():
    x = []
    y = []
    n = int(input("Введите количество точек (от 8 до 12)\n"))
    if n > 12 or n < 8:
        console_init()
    print("Теперь введите точки x и y через пробел")
    for i in range(n):
        a, b = input().split(" ")
        try:
            x.append(float(a))
            y.append(float(b))
        except ValueError:
            raise ValueError("Invalid value: x and y should be float numbers")

    action(x, y, n,"console")

