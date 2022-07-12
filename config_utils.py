import logging
import inspect
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from dataclasses import dataclass, asdict, fields


@dataclass
class BaseDataclass:

    def log_configurations(self, logger: logging.Logger):
        logger.info(f"Configurations for {type(self).__name__}")
        for configuration, value in self.as_dict().items():
            logger.info(f"\t{configuration} = {value}")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, env: dict) -> 'BaseDataclass':
        """Constructs a Config object from a dictionary.
        Allows extra arguments to be passed to the constructor."""
        # noinspection PyArgumentList
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


@dataclass
class LinearRegressionConfig(BaseDataclass):
    weight: float = round(np.random.uniform(low=0, high=1), 4)
    bias: float = round(np.random.uniform(low=-1, high=1), 4)

    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If the value of the field is none we can assign a value
            if getattr(self, field.name) is None:
                setattr(self, field.name, field.default)


@dataclass
class DataConfig(BaseDataclass):
    data_size: int
    mu: float = 0.0
    sigma: float = 1.0
    x_limits: Tuple[float, float] = (1., 10.)
    seed: Optional[int] = None

    @property
    def x_range(self) -> float:
        return self.x_limits[1] - self.x_limits[0]


@dataclass
class OutputConfig(BaseDataclass):
    output_dir: Path
    deviations_histogram: bool = False
    plot_line: bool = False
