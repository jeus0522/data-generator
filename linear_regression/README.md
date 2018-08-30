# Linear regression data generator

Create data points that are randomly deviated from a linear function. The deviations
come from a normal distribution that is configured in the parameters. The points
are uniformly distributed on the x axis in the specified range. The data points
are plotted to a graph. The generated data is saved to a .csv file with headers.

## Usage

The `linear_regression.py` script can be used on the console simply by:

```bash
python linear_regression.py
```

This will use the default values for all the configurable parameters.
To see a full list of available arguments use:

```bash
python linear_regression.py -h
```

## Example plot

The final plot of the data set generated will look something like this:

![linear_regression_data_example](plot_example.png)
