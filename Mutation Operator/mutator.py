import numpy as np


class Mutator:
    def __init__(self, pm:float = 0.1):
        self.pm = pm

    def do_mutation(self, coords: np.ndarray) -> np.ndarray:
        pass


