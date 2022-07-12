from pathlib import Path
from tkinter import *
from tkinter import ttk
from typing import Tuple

import numpy as np

from config_utils import DataConfig, LinearRegressionConfig, OutputConfig
from linear_regression import linear_data_job


class JobFrame:
    JOB_TYPES = ["linear_regression"]

    def __init__(self, app_frame: ttk.Frame):
        self.frame = ttk.Frame(app_frame)
        self.frame.grid()

        ttk.Label(self.frame, text="Job type:").grid(column=0, row=0)

        self.job_type_input = StringVar()
        self.job_type_input.set(self.JOB_TYPES[0])

        OptionMenu(self.frame, self.job_type_input, *self.JOB_TYPES).grid(column=1, row=0)

        # Default values
        job_config = LinearRegressionConfig()

        ttk.Label(self.frame, text="Weight:").grid(column=0, row=1)
        self.weight_input = ttk.Entry(self.frame)
        self.weight_input.insert(0, job_config.weight)
        self.weight_input.grid(column=1, row=1)
        ttk.Button(self.frame, text="Random",
                   command=lambda: self.random_weight()) \
            .grid(column=2, row=1)

        ttk.Label(self.frame, text="Bias:").grid(column=0, row=2)
        self.bias_input = ttk.Entry(self.frame)
        self.bias_input.insert(0, job_config.bias)
        self.bias_input.grid(column=1, row=2)
        ttk.Button(self.frame, text="Random",
                   command=lambda: self.random_bias()) \
            .grid(column=2, row=2)

    @property
    def weight(self) -> float:
        return float(self.weight_input.get())

    @property
    def bias(self) -> float:
        return float(self.bias_input.get())

    def random_weight(self):
        self.weight_input.delete(0, END)
        self.weight_input.insert(0, round(np.random.uniform(low=0, high=1), 4))

    def random_bias(self):
        self.bias_input.delete(0, END)
        self.bias_input.insert(0, round(np.random.uniform(low=-1, high=1), 4))

    def get_job_config(self) -> LinearRegressionConfig:
        return LinearRegressionConfig(weight=self.weight, bias=self.bias)


class DataFrame:

    def __init__(self, app_frame: ttk.Frame):
        self.frame = ttk.Frame(app_frame)
        self.frame.grid()

        # Default config
        data_config = DataConfig(data_size=20, seed=0)

        ttk.Label(self.frame, text="Data size:").grid(column=0, row=0)
        self.data_size_input = ttk.Entry(self.frame)
        self.data_size_input.insert(0, data_config.data_size)
        self.data_size_input.grid(column=1, row=0)

        ttk.Label(self.frame, text="Mu:").grid(column=0, row=1)
        self.mu_input = ttk.Entry(self.frame)
        self.mu_input.insert(0, data_config.mu)
        self.mu_input.grid(column=1, row=1)

        ttk.Label(self.frame, text="Sigma:").grid(column=0, row=2)
        self.sigma_input = ttk.Entry(self.frame)
        self.sigma_input.insert(0, data_config.sigma)
        self.sigma_input.grid(column=1, row=2)

        ttk.Label(self.frame, text="X limits:").grid(column=0, row=3)
        self.x_min_input = ttk.Entry(self.frame)
        self.x_min_input.insert(0, data_config.x_limits[0])
        self.x_min_input.grid(column=1, row=3)
        self.x_max_input = ttk.Entry(self.frame)
        self.x_max_input.insert(0, data_config.x_limits[1])
        self.x_max_input.grid(column=2, row=3)

        ttk.Label(self.frame, text="Seed:").grid(column=0, row=4)
        self.seed_input = ttk.Entry(self.frame)
        self.seed_input.insert(0, data_config.seed)
        self.seed_input.grid(column=1, row=4)

    @property
    def data_size(self) -> int:
        return int(self.data_size_input.get())

    @property
    def mu(self) -> float:
        return float(self.mu_input.get())

    @property
    def sigma(self) -> float:
        return float(self.sigma_input.get())

    @property
    def x_limits(self) -> Tuple[float, float]:
        return float(self.x_min_input.get()), float(self.x_max_input.get())

    @property
    def seed(self) -> int:
        return int(self.seed_input.get())

    def get_data_config(self) -> DataConfig:
        return DataConfig(data_size=self.data_size, mu=self.mu, sigma=self.sigma,
                          x_limits=self.x_limits, seed=self.seed)


def generate_data(job_frame: JobFrame, data_frame: DataFrame):
    output_dir = Path(Path(__file__).parent, "../generated_data")
    output_config = OutputConfig(output_dir=output_dir, plot_line=True, deviations_histogram=True)

    linear_data_job(job_frame.get_job_config(), data_frame.get_data_config(), output_config)


def main():
    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Data generator").grid(column=0, row=0)
    job_frame = JobFrame(frm)
    data_frame = DataFrame(frm)
    ttk.Button(frm, text="Generate",
               command=lambda: generate_data(job_frame, data_frame)) \
        .grid(column=0, row=5)
    root.mainloop()


if __name__ == '__main__':
    main()
