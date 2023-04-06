import numpy as np


class PoissonClutter2d:
    """Poisson clutter model.

    The number of clutters, k, follows a poisson distribution.
    The k clutters are uniformaly spatially distributed.
    """

    def __init__(self, density, range_):
        self.dentity = density
        self.range_ = range_

    def arise(self, center):

        num_clutter = np.random.poisson(lam=self.dentity*(self.range_**2))
        if num_clutter == 0:
            return np.empty((0, 2))

        clutter = center + np.random.uniform(low=-self.range_,
                                             high=self.range_,
                                             size=(num_clutter, 2))
        return clutter
