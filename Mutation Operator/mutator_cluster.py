from mutators.mutator import Mutator
import numpy as np
import random


class Cluster(Mutator):
    def do_mutation(self, coords: np.ndarray) -> np.ndarray:
        sz = coords.shape[0]
        mutrows = random.sample(range(0, sz), int(sz * self.pm))
        c = np.random.uniform(size=2)
        sdev = random.uniform(0.001, 0.01)
        norm = np.array([sdev, sdev])
        ncds = np.diag(norm)
        ncds = np.random.multivariate_normal(c, ncds, len(mutrows))
        coords[mutrows] = ncds
        for i in range(len(coords)):
            for j in range(len(coords[i])):
                if coords[i][j] <= 0 or coords[i][j] >= 1:
                    coords[i][j] = np.random.random()
        return coords
