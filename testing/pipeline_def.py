from abc import ABC, abstractmethod
import numpy as np

class Pipeline(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self, image : np.ndarray) -> str:
        # note that the image is already converted into the numpy format
        pass


