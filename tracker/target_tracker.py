from abc import abstractmethod, ABC

import numpy as np
import scipy.stats as stats

from filter import KalmanFilter
from utils import inner_elipsoide_data


class BayesianTargetTracker(KalmanFilter):
    """Bayesian target tracker."""

    def __init__(self, kalman_behavior, PG, PD,
                 parametric=False, clutter_density=None):

        super().__init__(kalman_behavior)

        self.PG = PG
        self.PD = PD
        self.gate_thresh = stats.chi2.ppf(q=PG, df=self.z_prior.shape[0])

        self.parametric = parametric
        self.clutter_density = clutter_density

        self.associate_proba = None
        self.associate_weight = None

    def extract_measurement(self, z):
        """Extract valid measurements."""

        return inner_elipsoide_data(z, self.z_prior, self.Pz_prior, self.gate_thresh)

    def update_posterior(self, z):
        """Update x[t|t] and P[t|t]."""

        if z.shape[0] == 0:
            # if measurements is not detected,
            # return the prior as the posterior.
            self.x_post, self.P_post = np.copy(self.x_prior), np.copy(self.P_prior)
            return

        self.associate_proba, self.associate_weight = \
            self.compute_proba(z, self.PG, self.PD, self.gate_thresh,
                               self.parametric, self.clutter_density)
        self.x_post, self.P_post = self.associate(self.associate_proba, z)

    def estimate(self, t, z, u_prev=0):
        """Estimate the state."""

        self.update_prior(t-1, u_prev)

        self.update_z_prior(t)

        z_valid = self.extract_measurement(z)

        self.update_posterior(z_valid)

        return self.x_post, self.P_post
