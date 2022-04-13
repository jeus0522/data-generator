import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config_utils import LinearRegressionConfig, DataConfig, OutputConfig


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


def generate_linear_data(job_config: LinearRegressionConfig,
                         data_config: DataConfig, output_config: OutputConfig):
    """
    Create data points that are randomly deviated from a linear function. The deviations
    come from a normal distribution that is configured in the parameters. The points
    are uniformly distributed on the x axis in the specified range. The data points
    are plotted to a graph. The generated data is saved to a .csv file with headers.

    :return: None
    """

    # Fixing random state for reproducibility
    if data_config.seed:
        np.random.seed(data_config.seed)

    # Create the graph figures
    if output_config.deviations_histogram:
        plt.figure(figsize=(10, 5))
        # Data subplot
        plt.subplot(121)
    else:
        plt.figure(figsize=(5, 5))

    # Create x values of the data points
    x = []
    x_range = data_config.x_limits[1] - data_config.x_limits[0]
    for i in range(data_config.data_size):
        x.append(data_config.x_limits[0] + (x_range / data_config.data_size) * i)

    print("Line weight = {}".format(job_config.weight))
    print("Line bias = {}".format(job_config.bias))

    # Generate random deviations from a normal distribution N(mu, sigma^2)
    deviations = np.random.randn(data_config.data_size) * data_config.sigma + data_config.mu

    # Create y values of the data points as y = (weight * x + bias) + deviation
    prediction = [value * job_config.weight + job_config.bias for value in x]
    y = prediction + deviations

    # Plot the data
    plt.axis("equal")
    plt.title("Data points")
    plt.plot(x, y, 'o', markersize=5)

    # Plot the line with the data points
    if output_config.plot_line:
        line_extra_space = 0.25
        x_line = []
        line_x_min = data_config.x_limits[0] - x_range * line_extra_space
        line_x_max = data_config.x_limits[1] + x_range * line_extra_space
        line_range = line_x_max - line_x_min
        for i in range(data_config.data_size):
            x_line.append(line_x_min + (line_range / data_config.data_size) * i)

        y_line = [value * job_config.weight + job_config.bias for value in x_line]

        if job_config.bias < 0:
            bias_text = " - {}".format(abs(round(job_config.bias, 3)))
        else:
            bias_text = " + {}".format(round(job_config.bias, 3))

        plt.text(x_line[0], y_line[-1],
                 "Linear function:\n"
                 "x * {}".format(round(job_config.weight, 3), round(job_config.bias, 3)) + bias_text,
                 fontweight='bold',
                 bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        plt.plot(x_line, y_line)
    else:
        # Give the data graph extra space
        extra_space = 0.5
        plt.xlim((data_config.x_limits[0] - x_range * extra_space,
                  data_config.x_limits[1] + x_range * extra_space))
        y_range = max(y) - min(y)
        plt.ylim((min(y) - y_range * extra_space, max(y) + y_range * extra_space))

    plt.grid()

    # Plot the histogram from where the deviations where obtained
    if output_config.deviations_histogram:
        plt.subplot(122)

        data = np.random.randn(100000) * data_config.sigma + data_config.mu

        plt.hist(data, bins=35)
        plt.title("Deviations histogram")

        plt.text(data_config.sigma * -4, 0.08 * data.shape[0],
                 "Normal distribution:\n"
                 "mu = {}\nsigma = {}".format(round(data_config.mu, 3), round(data_config.sigma, 3)),
                 fontweight='bold', fontsize=8,
                 bbox={"facecolor": "blue", "alpha": 0.5, "pad": 5})

        plt.grid()

    # Delete old data file
    if output_config.output_dir is None:
        output_config.output_dir = Path(Path(__file__).parent, "generated_data")

    if not output_config.output_dir.exists():
        output_config.output_dir.mkdir(parents=True)

    data_file = Path(output_config.output_dir, "linear_data.csv")
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
    parser = argparse.ArgumentParser(description="Create data points that are randomly deviated from "
                                                 "a linear function.")

    parser.add_argument("--data_size", type=int, default=10,
                        help="The number of data points to generate. Defaults to 10.")
    parser.add_argument("--x_limits", type=float, nargs=2, default=[1, 10],
                        help="The range of the x values for the generated data points. "
                             "Defaults to '1 10' (range [1, 10]).")
    parser.add_argument("--weight", type=float, default=None,
                        help="The 'w' parameter of the linear function. Defaults to random in [0, 1).")
    parser.add_argument("--bias", type=float, default=None,
                        help="The 'b' parameter of the linear function. Defaults to random in [-1, 1).")
    parser.add_argument("--mu", type=float, default=0.,
                        help="The mean of the normal distribution from where the deviations are generated.")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="The standard deviation of the normal distribution from where the "
                             "deviations are generated.")
    parser.add_argument("--deviations_histogram", default=False, action="store_true",
                        help="Activate to plot the deviations histogram.")
    parser.add_argument("--plot_line", default=False, action="store_true",
                        help="Activate to plot the linear function with the data points.")
    parser.add_argument("--seed", type=int, default=None,
                        help="The Numpy seed to use on the random numbers generation.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save the generated data file. Defaults to generated_data/ .")
    args = vars(parser.parse_args())

    if args["output_dir"] is None:
        output_dir = Path(Path(__file__).parent, "generated_data")
    else:
        output_dir = Path(args["output"])

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    args["output_dir"] = output_dir

    generate_linear_data(job_config=LinearRegressionConfig.from_dict(args),
                         data_config=DataConfig.from_dict(args),
                         output_config=OutputConfig.from_dict(args))


if __name__ == '__main__':
    main()
