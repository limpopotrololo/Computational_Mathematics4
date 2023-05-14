from console_init import *
from file_init import *

def start():
    io = int(input("Введите 1 для консольного ввода \nВведите 2 для файлового ввода\n"))
    if io == 1:
        console_init()
    elif io == 2:
        file_init()
    else:
        print("Данные некорректны")
        start()




print("program is staring")
start()