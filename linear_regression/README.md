# Linear regression data generator

Create data points that are randomly deviated from a linear function. The deviations
come from a normal distribution that is configured in the parameters. The points
are uniformly distributed on the x axis in the specified range. The data points
can be plotted to a graph. The generated data is saved to a .csv file with headers.

Example plot:
![linear_regression_data_example](plot_example.png)


## Usage

### Script from console

The `linear_regression.py` script can be used on the console simply by:

```bash
python linear_regression.py
```

This will use the default values for all the configurable parameters.
To see a full list of available arguments use:

```bash
python linear_regression.py -h
```

### Streamlit APP
The Streamlit UI is served [here](https://fiquinho-data-generator-data-generator-app-yo8n6m.streamlitapp.com/Linear_Regression).

If you want to run it locally you can use the **Data_generator_APP** script 
from the project root with:

```bash
python -m streamlit run Data_generator_APP.py
```

### Tkinter UI

There is also a Tkinter UI to run generation jobs and set the parameters. 
To use ir run:

```bash
python ui.py -h
```
