# coding: utf-8
from and_gate import AND
from or_gate import OR
from nand_gate import NAND
import matplotlib.pyplot as plt


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == "__main__":
    ys = []
    xs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for x in xs:
        y = XOR(x[0], x[1])
        ys.append(y)
        print(str(x) + " -> " + str(y))

    # Plotting
    for i, point in enumerate(xs):
        if ys[i] == 0:
            plt.scatter(point[0], point[1], marker="*", s=100)  # Star for 0
        else:
            plt.scatter(point[0], point[1], marker="o", s=100)  # Circle for 1

    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("XOR Gate Output")
    plt.grid(True)
    plt.savefig("xor.png")
