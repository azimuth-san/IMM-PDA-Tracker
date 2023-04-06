from abc import abstractclassmethod, ABC
import numpy as np
import numpy.linalg as la

from filter import KalmanFilter
from utils import gaussian_mixture_moment


class AssociateMixin(ABC):
    """Association class."""

    def compute_proba(self: KalmanFilter, z: np.array,
                      PG: float, PD: float, gamma: float,
                      parametric: float = False, clutter_density: float = None):
        """Compute asscocation probability."""

        proba = np.zeros(z.shape[0] + 1)
        weight = np.zeros(z.shape[0] + 1)
        inv_Pz = la.inv(self.Pz_prior)

        mu = z - self.z_prior
        rhs = (inv_Pz @ mu.T).T
        e = np.exp(-0.5 * np.sum(mu * rhs, axis=1))

        if parametric:
            det_Pz = la.det(2 * np.pi * self.Pz_prior)
            PDG = PD * PG
            b = clutter_density * np.sqrt(det_Pz) * (1 - PDG) / PD

        else:
            num_z, dim_z = z.shape[0], z.shape[1]

            if dim_z == 1:
                cz = 2
            elif dim_z == 2:
                cz = np.pi
            elif dim_z == 3:
                cz = 4 * np.pi / 3
            else:
                raise NotImplementedError

            PDG = PD * PG
            b = np.sqrt((2 * np.pi / gamma) ** dim_z) * num_z * (1 - PDG) / (cz * PD)

        weight[0] = b
        weight[1:] = e
        proba = weight / np.sum(weight)

        return proba, weight

    @abstractclassmethod
    def associate(self: KalmanFilter, proba: np.array, z: np.array):
        """Associate measurements with the kamnan filter."""
        pass


class PruneAssociate(AssociateMixin):
    def associate(self: KalmanFilter, proba: np.array, z: np.array):
        """Associate the nearest measurement with the kalman filter.
        Prune posterior distributions against other measurements.
        """

        w_max_idx = np.argmax(proba)
        if w_max_idx == 0:
            return self.x_prior, self.P_prior

        # select the nearest measurement (prune the other measurements)
        z_nearest = z[w_max_idx-1, :]

        x_post, P_post = self.behavior.compute_posterior(
                                    self.x_prior, self.P_prior,
                                    z_nearest, self.z_prior, self.Pz_prior)

        return x_post, P_post


class MergeAssociate(AssociateMixin):
    def associate(self: KalmanFilter, proba: np.array, z: np.array):
        """Associate the measurements with the kalman filter.
        Merge posterior distributions against each measurement.
        """

        # estimate each posterior distribution and merge.
        x_post_each = []
        P_post_each = []
        for i in range(z.shape[0] + 1):
            if i == 0:
                x, P = self.x_prior, self.P_prior
            else:
                x, P = self.behavior.compute_posterior(
                                        self.x_prior, self.P_prior,
                                        z[i-1, :], self.z_prior, self.Pz_prior)
            x_post_each.append(x)
            P_post_each.append(P)

        x_post, P_post = gaussian_mixture_moment(
                            x_post_each, P_post_each, proba)
        return x_post, P_post
