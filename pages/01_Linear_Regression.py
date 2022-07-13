import base64

import numpy as np
import pandas as pd
import streamlit as st

from config_utils import DataConfig, LinearRegressionConfig
from linear_regression.linear_regression import deviations_histogram, data_plot, generate_linear_data


def random_weight():
    st.session_state['weight'] = np.random.uniform(low=-1, high=1)


def random_bias():
    st.session_state['bias'] = np.random.uniform(low=-1, high=1)


if 'weight' not in st.session_state:
    random_weight()
if 'bias' not in st.session_state:
    random_bias()


st.markdown("# Linear Regression")

data_size = st.number_input("Data size", value=20, min_value=1, help="Number of datapoint to generate")
data_size = int(data_size)

c1, c2 = st.columns([1, 1])
with c1:
    min_x = st.number_input("Min X", help="The minimum value possible for X")
with c2:
    max_x = st.number_input("Max X", value=10., help="The maximum value possible for X")

if max_x <= min_x:
    st.error(f"Min X ({min_x:.3f}) needs to be smaller than Max X ({max_x:.3f})")

st.markdown("## Deviations")
st.markdown("The deviations are generated from a normal distribution with:")
mu = st.number_input("Mean (\u03BC)", value=0., help="The mean of the distribution")
sigma = st.number_input("Standard deviation (\u03C3)", value=1., help="The variance of the distribution")

deviations_fig = deviations_histogram(mu, sigma)
with st.expander("See normal distribution"):
    st.write(deviations_fig)

st.markdown("## Linear function")

c1, c2 = st.columns([8, 2])
with c1:
    weight = st.number_input("Weight", value=st.session_state["weight"],
                             help="The weight of linear function", key="weight_input")
    bias = st.number_input("Bias", value=st.session_state["bias"],
                           help="The bias of linear function", key="bias_input")

with c2:
    st.write("")
    st.write("")

    random_weight_button = st.button("Random", key="random_weight_button", on_click=random_weight)
    st.write("")
    st.write("")

    random_bias_button = st.button("Random", key="random_bias_button", on_click=random_bias)


plot_line = st.checkbox("Plot line", value=True)

st.markdown("## Explore the data")
data_config = DataConfig(data_size, mu, sigma, (min_x, max_x), seed=None)
job_config = LinearRegressionConfig(weight, bias)
x, y = generate_linear_data(job_config, data_config)

data_fig = data_plot(x, y, data_config, job_config, plot_line=plot_line)
st.write(data_fig)


def download_button(object_to_download, download_filename, button_text):
    """From: https://github.com/streamlit/example-app-csv-wrangler"""

    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as _:
        b64 = base64.b64encode(object_to_download).decode()

    button_id = "download_button"

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
            custom_css
            + f'<a download="{download_filename}" id="{button_id}" '
              f'href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )

    st.markdown(dl_link, unsafe_allow_html=True)


df = pd.DataFrame(x, columns=["x"])
df['y'] = y.tolist()
st.dataframe(df)

st.markdown("## Download data")
download_button(df, "linear_regression_data.csv", "Download to CSV")
