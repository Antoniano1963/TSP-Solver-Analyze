from mutators.mutator import Mutator
import numpy as np
import random


def genLinearFunc():
    b0 = random.uniform(0, 1)
    if (b0 < 0.5):
        b1 = random.uniform(0, 3)
    else:
        b1 = random.uniform(-3, 0)

    def func(x):
        return b0 + b1 * x

    return func


class LinearProject(Mutator):
    def __init__(self, pm:float = 0.1, jt:float=0, sdev:float=0):
        super(LinearProject, self).__init__(pm)
        self.jt = jt
        self.sdev = sdev

    def do_mutation(self, coords: np.ndarray) -> np.ndarray:
        sz = coords.shape[0]
        mutrows = random.sample(range(0, sz), int(sz * self.pm))

        linearFunc = genLinearFunc()

        coords[mutrows, 1] = list(map(linearFunc, coords[mutrows, 0]))

        if random.uniform(0, 1) < self.jt and self.sdev > 0:
            coords[mutrows, 1] += np.random.normal(size=sz, scale=self.sdev)

        for i in range(len(coords)):
            for j in range(len(coords[i])):
                if coords[i][j] <= 0 or coords[i][j] >= 1:
                    coords[i][j] = np.random.random()

        return coords
