import matplotlib.pyplot as plt


def plot_approximation(x, y, y_aprox,name):
    plt.plot(x, y, 'ro', label='Original points')
    plt.plot(x, y_aprox, 'b-', label=name)
    plt.legend()
    plt.show()

