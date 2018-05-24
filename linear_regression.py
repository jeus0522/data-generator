import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Fixing random state for reproducibility
np.random.seed(19680801)


def linear_function(x: np.array, w: float=None, b: float=None) -> float:

    y = w * x + b
    return y


def generate_linear_data(data_size: int=20, mu: float=0, sigma: float=1.0,
                         weight: float=None, bias: float=None,
                         x_limits: tuple=None, deviations_histogram: bool=False,
                         plot_line: bool=False):

    # Create the graph figures
    if deviations_histogram:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(figsize=(5, 5))

    # Data plot
    if deviations_histogram:
        plt.subplot(121)

    # x values operations
    if x_limits is None:
        x_limits = (1, 10)

    x_range = x_limits[1] - x_limits[0]

    # Create x values of the data points
    x = []
    for i in range(data_size):
        x.append(x_limits[0] + (x_range / data_size) * i)

    # Weight and bias generation
    if weight is None:
        weight = np.random.uniform(low=0, high=1)
    if bias is None:
        bias = np.random.uniform(low=-1, high=1)

    print("Weight = {}".format(weight))
    print("Bias = {}".format(bias))

    # Generate random deviations from a normal distribution N(mu, sigma^2)
    deviations = np.random.randn(data_size) * sigma + mu

    # Create y values of the data points as y = (weight * x + bias) + deviation
    prediction = [value * weight + bias for value in x]
    y = prediction + deviations

    plt.plot(x, y, 'o')

    if plot_line:
        # TODO: Plot weight and bias to the graph
        line_extra_space = 0.25
        x_line = []
        line_x_min = x_limits[0] - x_range * line_extra_space
        line_x_max = x_limits[1] + x_range * line_extra_space
        line_range = line_x_max - line_x_min
        for i in range(data_size):
            x_line.append(line_x_min + (line_range / data_size) * i )

        y_line = [value * weight + bias for value in x_line]
        plt.plot(x_line, y_line)
    else:
        extra_space = 0.5
        plt.xlim((x_limits[0] - x_range * extra_space, x_limits[1] + x_range * extra_space))
        y_range = max(y) - min(y)
        plt.ylim((min(y) - y_range * extra_space, max(y) + y_range * extra_space))


    plt.grid()

    if deviations_histogram:
        # TODO: Plot mu and sigma to the graph
        plt.subplot(122)
        data = np.random.randn(100000) * sigma + mu
        plt.hist(data, bins=35)
        plt.grid()

    # TODO: Continue from here
    data_file = Path("data.csv")
    if data_file.exists():
        data_file.unlink()

    with open("data.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["x", "y"])
        for i in tqdm(range(len(x))):
            writer.writerow([x[i], y[i]])



    plt.show()


def main():

    generate_linear_data(data_size=10, deviations_histogram=True)

if __name__ == '__main__':
    main()
