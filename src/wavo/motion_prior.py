import abc

import numpy as np


class BaseMotionPrior:

    @abc.abstractmethod
    def predict(self, dt: float):
        pass

    @abc.abstractmethod
    def update(self, T: np.ndarray, dt: float):
        pass
