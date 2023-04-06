import numpy as np
import numpy.linalg as la
from typing import List


def maharanobis_distance(z: np.array, zc: np.array, inv_covar: np.array):

    # z.shape: (num, dimension)
    dz = z - zc
    rhs = (inv_covar @ dz.T).T
    distance = np.sum(dz * rhs, axis=1)
    return distance


def inner_elipsoide_data(x: np.array, center: np.array, covar: np.array,
                         thresh: float):

    # x.shape: (num, dimension)
    if x.shape[0] == 0:
        return np.empty_like(x)

    inv_covar = la.inv(covar)
    distances = maharanobis_distance(x, center, inv_covar)

    mask = distances <= thresh
    return x[mask]


def gaussian_mixture_moment(xs: List[np.array], Ps: List[np.array],
                            weights: np.array):

    x_mix = np.zeros_like(xs[0])
    for i in range(len(xs)):
        x_mix += weights[i] * xs[i]

    P_mix = np.zeros_like(Ps[0])
    for i in range(len(xs)):
        delta = (x_mix - xs[i])[:, np.newaxis]
        spread = delta @ delta.T
        P_mix += weights[i] * (Ps[i] + spread)

    return x_mix, P_mix
