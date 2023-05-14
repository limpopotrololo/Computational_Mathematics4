from manager import action


def file_init():
    try:
        with open('input.txt', 'r') as file:
            n = int(file.readline().strip())
            if 8 <= n < 12:
                x = []
                y = []
                for i in range(n):
                    line = file.readline().strip().split()
                    try:
                        x.append(float(line[0]))
                        y.append(float(line[1]))
                    except ValueError:
                        raise ValueError("Invalid value: x and y should be float numbers")

                action(x, y, n, "file")
            else:
                raise ValueError("Invalid number of items: n should be greater or equal to 8 and less than 12")

    except ValueError as e:
        print("Error:", e)



