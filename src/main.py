import matplotlib.pyplot as plt
import numpy as np

import example 

def generate_data(n=256, m=3, c=2, plot=True):
    x = np.random.rand(n)
    noise = np.random.randn(n)/4

    y = m * x + c + noise

    if plot==True:
        plt.scatter(x, y)
        plt.show()

    return x, y


def main():
    x, y = generate_data()


if __name__=="__main__":
    main()


