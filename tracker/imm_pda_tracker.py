import numpy as np
import numpy.linalg as la

from filter import IMMFilter
from utils import gaussian_mixture_moment
from utils import inner_elipsoide_data


class IMMPDATracker(IMMFilter):
    """IMM PDA Tracker."""

    def __init__(self, pda_trackers, mode_proba, transition_mat,
                 valid_region='mixture'):

        super().__init__(pda_trackers, mode_proba, transition_mat)

        self.z_prior = np.zeros_like(self.kalman_filters[0].z_prior)
        self.Pz_prior = np.zeros_like(self.kalman_filters[0].Pz_prior)

        # valid_region_type is 'mixture' or 'max_det'.
        # Tracking by 'max_det' is a bit unstable in this implementation.
        self.valid_region_type = valid_region
        self.valid_region_idx = 0

    def extract_measurement(self, z):
        """Extract measurements."""

        if self.valid_region_type == 'max_det':
            dets = [np.linalg.det(pdaf.Pz_prior)
                    for pdaf in self.kalman_filters]

            self.valid_region_idx = np.argmax(dets)
            pdaf = self.kalman_filters[self.valid_region_idx]
            # print(f'\ndets : {dets}')
            self.z_prior, self.Pz_prior = np.copy(pdaf.z_prior), np.copy(pdaf.Pz_prior)
            self.z_valid = inner_elipsoide_data(
                            z, pdaf.z_prior, pdaf.Pz_prior, pdaf.gate_thresh)

        elif self.valid_region_type == 'mixture':

            zs = [pdaf.z_prior for pdaf in self.kalman_filters]
            Pzs = [pdaf.Pz_prior for pdaf in self.kalman_filters]
            self.z_prior, self.Pz_prior = gaussian_mixture_moment(
                                            zs, Pzs, self.predicted_mode_proba)
            # dets = [np.linalg.det(pdaf.Pz_prior)
            #         for pdaf in self.kalman_filters]
            # dets.append(np.linalg.det(self.Pz_prior))
            # print(f'\nproba : {self.predicted_mode_proba}')
            # print(f'dets : {dets}')
            pdaf = self.kalman_filters[0]
            self.z_valid = inner_elipsoide_data(
                            z, self.z_prior, self.Pz_prior, pdaf.gate_thresh)
        # print(f'num_z : {self.z_valid.shape[0]}')

    def estimate_each(self, t, z, u_prev):
        """Estimate each PDA filter's state."""

        for pdaf in self.kalman_filters:
            pdaf.update_prior(t-1, u_prev)
            pdaf.update_z_prior(t)

        self.extract_measurement(z)

        for pdaf in self.kalman_filters:
            pdaf.update_posterior(self.z_valid)

    def update_likelihood(self, z):
        """Update the likelihoods."""

        num_z, dim_z = z.shape[0], z.shape[1]

        if dim_z == 1:
            c = 2
        elif dim_z == 2:
            c = np.pi
        elif dim_z == 3:
            c = 4 * np.pi / 3
        else:
            raise NotImplementedError

        if self.valid_region_type == 'max_det':
            pdaf = self.kalman_filters[self.valid_region_idx]
            volume = c * np.sqrt(la.det(pdaf.gate_thresh * pdaf.Pz_prior))

        elif self.valid_region_type == 'mixture':
            pdaf = self.kalman_filters[0]
            volume = c * np.sqrt(la.det(pdaf.gate_thresh * self.Pz_prior))

        alpha = (1 - pdaf.PD * pdaf.PG) / (volume ** num_z)
        beta = pdaf.PD * (volume ** (1 - num_z))

        for j in range(self.num_f):

            pdaf = self.kalman_filters[j]

            if num_z == 0:
                self.likelihood[j] = alpha

            elif num_z >= 1:
                det_S = la.det(2 * np.pi * pdaf.Pz_prior)
                e = pdaf.associate_weight[1:]
                self.likelihood[j] = alpha + (beta / (num_z * np.sqrt(det_S))) * np.sum(e)

    @property
    def gate_thresh(self):

        if self.valid_region_type == 'max_det':
            return self.kalman_filters[self.valid_region_idx].gate_thresh

        elif self.valid_region_type == 'mixture':
            return self.kalman_filters[0].gate_thresh

    def estimate(self, t, z, u_prev):
        """Estimate the state."""

        self.update_mixing_probability()

        self.mixing()

        self.estimate_each(t, z, u_prev)

        self.update_likelihood(self.z_valid)

        self.update_mode_probability()

        self.update_posterior()

        return self.x_post, self.P_post
