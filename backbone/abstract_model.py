"""This module contains the abstract Model class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import logging

# Initialize the logger.
logging.basicConfig(level=logging.INFO)


class Model(ABC):
    """An abstract class representing a model object.

    Algorithm-specific model implementations must inherit from this class.
    """

    @abstractmethod
    def __init__(self, is_verbose: bool = False, cores: int = 1) -> None:
        """Initialize the model."""
        self.is_trained: bool = False
        self.is_verbose: bool = is_verbose
        self.cores: int = cores

    @abstractmethod
    def train(self, train_data: Any) -> None:
        """Trains the object with the given dataset."""
        pass

    @abstractmethod
    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        """Generates predictions for the given dataset."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the model."""
        pass