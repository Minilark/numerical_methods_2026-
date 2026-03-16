import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):

    x = []
    y = []

    with open(filename) as file:

        reader = csv.DictReader(file)

        for row in reader:

            x.append(float(row["Objects"]))
            y.append(float(row["FPS"]))

    return np.array(x), np.array(y)


def divided_differences(x, y):

    n = len(x)

    table = np.zeros((n, n))
    table[:,0] = y

    for j in range(1,n):

        for i in range(n-j):

            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])

    return table


def newton(x, table, value):

    n = len(x)

    result = table[0][0]
    mult = 1

    for i in range(1,n):

        mult *= (value - x[i-1])
        result += table[0][i] * mult

    return result


def factorial_interpolation(x, y, value):

    h = x[1] - x[0]

    diff = [y.copy()]

    for i in range(1,len(y)):

        diff.append(np.diff(diff[i-1]))

    s = (value - x[0]) / h

    result = y[0]
    term = 1

    for i in range(1,len(y)):

        term *= (s-(i-1))/i
        result += term * diff[i][0]

    return result


def find_limit(x, table):

    for n in range(100,2000):

        if newton(x, table, n) <= 60:

            return n


def plot_interpolation(x, y, table):

    xs = np.linspace(min(x), max(x), 300)

    ys = [newton(x, table, i) for i in xs]

    plt.figure()

    plt.scatter(x, y)
    plt.plot(xs, ys)

    plt.title("FPS(n)")
    plt.xlabel("Objects")
    plt.ylabel("FPS")

    plt.show()


def node_research(x, y):

    nodes = [5,10,20]
    errors = []

    real = np.linspace(min(x), max(x), 200)

    for n in nodes:

        xs = np.linspace(min(x), max(x), n)
        ys = np.interp(xs, x, y)

        table = divided_differences(xs, ys)

        err = 0

        for val in real:

            p = newton(xs, table, val)

            real_val = np.interp(val, x, y)

            err += abs(p-real_val)

        errors.append(err/len(real))

    plt.figure()

    plt.plot(nodes, errors, marker="o")

    plt.title("Error vs nodes")
    plt.xlabel("Nodes")
    plt.ylabel("Average error")

    plt.show()


def runge_effect(x, y):

    plt.figure()

    real_x = np.linspace(min(x), max(x), 300)

    for n in [5,10,20]:

        xs = np.linspace(min(x), max(x), n)
        ys = np.interp(xs, x, y)

        table = divided_differences(xs, ys)

        approx = [newton(xs, table, i) for i in real_x]

        plt.plot(real_x, approx, label=f"{n} nodes")

    plt.scatter(x, y)

    plt.title("Runge effect")
    plt.xlabel("Objects")
    plt.ylabel("FPS")

    plt.legend()

    plt.show()



x,y = read_data("data.csv")

print("X:",x)
print("Y:",y)

table = divided_differences(x,y)

print("\nTable of divided differences")
print(table)


fps1000_newton = newton(x, table, 1000)
fps1000_fact = factorial_interpolation(x, y, 1000)

print("\nFPS for 1000 objects (Newton):", fps1000_newton)
print("FPS for 1000 objects (Factorial):", fps1000_fact)


limit = find_limit(x, table)

print("Minimum objects for FPS≈60:", limit)


plot_interpolation(x,y,table)

node_research(x,y)

runge_effect(x,y)