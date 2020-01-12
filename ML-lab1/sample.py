import numpy as np
import matplotlib.pyplot as plt
import random

def sample(dataAmount):
    x = np.linspace(0, 1, dataAmount)
    y = np.sin(2*np.pi*x)
    miu = 0
    sigma = 0.05
    fname = str(dataAmount)+".txt"
    f = open(fname, "w+")
    for i in range(x.size):
        x[i] += random.gauss(miu, sigma)
        y[i] += random.gauss(miu, sigma)
        f.write(str(x[i]))
        f.write("\t")
        f.write(str(y[i]))
        f.write("\n")
    f.close()
    plt.scatter(x, y, marker=".")
    plt.show()
sample(100)