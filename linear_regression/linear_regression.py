import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def linear_function(x: np.array, w: float=None, b: float=None) -> np.array:
    """
    Apply a linear function to all the values in a numpy array.

    :param x: A list of values to apply the function
    :param w: The slope of the linear function
    :param b: Intersection point of the linear function with the y axis
    :return: A list of values
    """

    y = w * x + b
    return y


def generate_linear_data(data_size: int=20, mu: float=0, sigma: float=1.0,
                         weight: float=None, bias: float=None,
                         x_limits: tuple=(1, 10), deviations_histogram: bool=False,
                         plot_line: bool=False, seed: int=None, output: Path=None):
    """
    Create data points that are randomly deviated from a linear function. The deviations
    come from a normal distribution that is configured in the parameters. The points
    are uniformly distributed on the x axis in the specified range. The data points
    are plotted to a graph. The generated data is saved to a .csv file with headers.

    :param data_size: The number of data points to create.
    :param mu: The mean of the deviations normal distribution.
    :param sigma: The standard deviation of the deviations normal distribution.
    :param weight: The slope of the linear function.
    :param bias: Intersection point of the linear function with the y axis.
    :param x_limits: A tuple with the range of the generated x values.
    :param deviations_histogram: Activate to plot the deviations histogram.
    :param plot_line: Activate to plot the linear function with the data points.
    :param seed: The Numpy seed to use on the random numbers generation.
    :param output: Where to save the generated data file. Defaults to this script folder.
    :return: None
    """

    # Fixing random state for reproducibility
    if seed:
        np.random.seed(seed)

    if not weight:
        weight = np.random.uniform(low=0, high=1)
    if not bias:
        bias = np.random.uniform(low=-1, high=1)

    # Create the graph figures
    if deviations_histogram:
        plt.figure(figsize=(10, 5))
        # Data subplot
        plt.subplot(121)
    else:
        plt.figure(figsize=(5, 5))

    # Create x values of the data points
    x = []
    x_range = x_limits[1] - x_limits[0]
    for i in range(data_size):
        x.append(x_limits[0] + (x_range / data_size) * i)

    print("Line weight = {}".format(weight))
    print("Line bias = {}".format(bias))

    # Generate random deviations from a normal distribution N(mu, sigma^2)
    deviations = np.random.randn(data_size) * sigma + mu

    # Create y values of the data points as y = (weight * x + bias) + deviation
    prediction = [value * weight + bias for value in x]
    y = prediction + deviations

    # Plot the data
    plt.axis("equal")
    plt.title("Data points")
    plt.plot(x, y, 'o', markersize=5)

    # Plot the line with the data points
    if plot_line:
        line_extra_space = 0.25
        x_line = []
        line_x_min = x_limits[0] - x_range * line_extra_space
        line_x_max = x_limits[1] + x_range * line_extra_space
        line_range = line_x_max - line_x_min
        for i in range(data_size):
            x_line.append(line_x_min + (line_range / data_size) * i)

        y_line = [value * weight + bias for value in x_line]

        if bias < 0:
            bias_text = " - {}".format(abs(round(bias, 3)))
        else:
            bias_text = " + {}".format(round(bias, 3))

        plt.text(x_line[0], y_line[-1],
                 "Linear function:\n"
                 "x * {}".format(round(weight, 3), round(bias, 3)) + bias_text,
                 fontweight='bold',
                 bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        plt.plot(x_line, y_line)
    else:
        # Give the data graph extra space
        extra_space = 0.5
        plt.xlim((x_limits[0] - x_range * extra_space, x_limits[1] + x_range * extra_space))
        y_range = max(y) - min(y)
        plt.ylim((min(y) - y_range * extra_space, max(y) + y_range * extra_space))

    plt.grid()

    # Plot the histogram from where the deviations where obtained
    if deviations_histogram:
        plt.subplot(122)

        data = np.random.randn(100000) * sigma + mu

        plt.hist(data, bins=35)
        plt.title("Deviations histogram")

        plt.text(sigma * -4, 0.08 * data.shape[0],
                 "Normal distribution:\n"
                 "mu = {}\nsigma = {}".format(round(mu, 3), round(sigma, 3)),
                 fontweight='bold', fontsize=8,
                 bbox={"facecolor": "blue", "alpha": 0.5, "pad": 5})

        plt.grid()

    # Delete old data file
    if not output:
        output = Path(__file__).parent
    data_file = Path(output, "linear_data.csv")
    if data_file.exists():
        data_file.unlink()

    # Save generated data to a csv file
    with open(data_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["x", "y"])
        for i in tqdm(range(len(x))):
            writer.writerow([x[i], y[i]])

    plt.show()


def main():

    generate_linear_data(data_size=10, x_limits=(1, 10), sigma=1,
                         deviations_histogram=True, plot_line=True,
                         seed=19680801)


if __name__ == '__main__':
    main()
