import argparse
import os

import yaml
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as pat

from models import ConstantVelocity, CoordinatedTurn
from filter import LinearKalmanBehavior
from filter import ExtendedKalmanBehavior, UnscentedendKalmanBehavior
from tracker import PDATracker
from tracker import IMMPDATracker

from clutter import PoissonClutter2d
from utils import EllipseShape, Detector


def parse_arguments():
    """Parse optional arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfg/sim_imm_pda.yaml',
                        help='configuration file.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed of random variables.')
    parser.add_argument('--output_dir', default='result',
                        help='output directory.')
    return parser.parse_args()


def make_target_tracker(cfg):
    """Make an single target tracker."""

    to_model_cls = {
        'ConstantVelocity': ConstantVelocity,
        'CoordinatedTurn': CoordinatedTurn,
    }

    to_kf_behavior_cls = {
        'LKF': LinearKalmanBehavior,
        'EKF': ExtendedKalmanBehavior,
        'UKF': UnscentedendKalmanBehavior,
    }

    to_tracker_cls = {
        'pda': PDATracker,
    }

    trackers = []
    dt = cfg['sampling_period']
    param_models = cfg['dynamics']['model']

    is_imm = cfg['estimator']['IMM']
    param_kfs = cfg['estimator']['kalman_filter']

    tracker_cls = to_tracker_cls[cfg['associate']['kind']]
    PD = cfg['associate']['detection_proba']
    PG = cfg['associate']['gate_proba']
    is_parametric = cfg['associate']['parametric']
    clutter_density = cfg['clutter']['spatial_density']

    for prm_kf in param_kfs:

        idx = prm_kf['model_idx']
        prm_model = param_models[idx]

        model_cls = to_model_cls[prm_model['kind']]
        model = model_cls(dt=dt)

        covar_w = np.diag(np.array(prm_model['sigma_w'])**2)
        covar_v = np.diag(np.array(prm_model['sigma_v'])**2)
        kf_behavior_cls = to_kf_behavior_cls[prm_kf['kind']]
        kf_behavior = kf_behavior_cls(model, covar_w, covar_v)

        tracker = tracker_cls(kf_behavior, PG, PD,
                              is_parametric, clutter_density)
        tracker.init_posterior(P=np.diag(prm_kf['initial_P']))
        trackers.append(tracker)

    if is_imm:
        mode_proba = np.array(cfg['estimator']['mode_proba'])
        transition_mat = np.array(cfg['estimator']['transition_mat'])
        tracker = IMMPDATracker(trackers, mode_proba, transition_mat,
                                valid_region='mixture')
    else:
        tracker = trackers[0]

    return tracker


def extract_models(cfg):
    """Extract models and parameters."""

    to_model_cls = {
        'ConstantVelocity': ConstantVelocity,
        'CoordinatedTurn': CoordinatedTurn,
    }

    models, covar_ws, covar_vs, initial_xs = [], [], [], []

    for param in cfg['dynamics']['model']:

        cls = to_model_cls[param['kind']]
        models.append(cls(dt=cfg['sampling_period']))

        covar_ws.append(np.diag(np.array(param['sigma_w'])**2))
        covar_vs.append(np.diag(np.array(param['sigma_v'])**2))
        initial_xs.append(np.array(param['initial_x']))

    return models, covar_ws, covar_vs, initial_xs


def generate_target_data(cfg):
    """Generate target data."""

    def _generate_noise(covar):
        mean = np.zeros(covar.shape[0])
        return np.random.multivariate_normal(mean, covar)

    models, covar_ws, covar_vs, initial_xs = extract_models(cfg)

    steps = cfg['steps']
    change_interval = cfg['dynamics']['change_interval']

    ndim_x = max([m.NDIM['x'] for m in models])
    ndim_z = models[0].NDIM['z']  # outputs are same dimension.

    x_hist = np.zeros((steps, ndim_x))  # history
    z_hist = np.zeros((steps, ndim_z))
    modes = np.zeros(steps, dtype=np.int32)

    mode = cfg['dynamics']['initial_model_idx']
    initial_x = np.array(initial_xs[mode])
    x_hist[0, :initial_x.shape[0]] = initial_x
    model, covar_w, covar_v = models[mode], covar_ws[mode], covar_vs[mode]

    for t in range(steps):

        # change model
        if (t > 0) and (t % change_interval == 0):
            mode = (mode + 1) % len(models)
            model = models[mode]
            covar_w, covar_v = covar_ws[mode], covar_vs[mode]

        w = _generate_noise(covar_w)
        v = _generate_noise(covar_v)

        ndim_x = model.NDIM['x']

        if t > 0:
            x_hist[t, :ndim_x] = model.state_equation(
                                    t-1, x_hist[t-1, :ndim_x], 0, w)

        z = model.observation_equation(t, x_hist[t, :ndim_x], v)
        z_hist[t] = z

        modes[t] = mode

    return x_hist, z_hist, modes


def track_target(cfg, tracker, z_tgt_hist):
    """Track the target."""

    steps = cfg['steps']

    clutter_model = PoissonClutter2d(cfg['clutter']['spatial_density'],
                                     cfg['clutter']['range'])
    detector = Detector(cfg['associate']['detection_proba'])
    is_imm = cfg['estimator']['IMM']

    measurement_sets = []
    valid_regions = []

    # postriors to be estimated
    x_posts = np.zeros((steps, tracker.x_post.shape[0]))

    # priors to be predicted
    z_priors = np.zeros((steps, tracker.z_prior.shape[0]))

    if is_imm:
        mode_proba = np.zeros((steps, len(tracker.kalman_filters)))
    else:
        mode_proba = np.ones((steps, 1))

    # for t in range(10):
    for t in range(steps):

        print(f'\rstep : {t+1}/{steps}', end='')

        clutter = clutter_model.arise(z_tgt_hist[t])

        # set measurements
        if t == 0:
            measurements = np.expand_dims(z_tgt_hist[t], axis=0)
            labels = np.array([1], dtype=np.int32)
        else:
            measurements, labels = detector.detect(z_tgt_hist[t], clutter)

        measurement_sets.append([labels, measurements])

        # initialize
        if t == 0:
            if is_imm:
                x_ = np.array([measurements[0, 0], 0.,
                               measurements[0, 1], 0., 0.])
            else:
                x_ = np.array([measurements[0, 0], 0.,
                               measurements[0, 1], 0.])
            x_posts[t, :] = x_
            tracker.init_posterior(x=x_posts[t, :])
            valid_regions.append(None)

        # track the target.
        elif t > 0:
            x_posts[t], _ = tracker.estimate(t, measurements, 0)
            z_priors[t] = tracker.z_prior

            # ellipse : (z - z_hat).T @ S_inverse @ (z - z_hat) = thresh
            S_inverse = la.inv(tracker.Pz_prior)
            e = EllipseShape.from_quadratic_form(
                    S_inverse, tracker.gate_thresh, tracker.z_prior)
            valid_regions.append(e)

    return measurement_sets, x_posts, z_priors, valid_regions, mode_proba


def plot_trajectory(cfg, args,
                    target_states, target_measurements, clutters,
                    estimations, predictions, valid_regions):
    """Plot the tracking result."""

    title = f'Trajectory'
    if cfg['estimator']['IMM']:
        title = title + ' (IMM-PDAF)'
    else:
        title = title + ' (PDAF)'

    fig = plt.figure(1, (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)

    ax.plot(target_states[:, 0], target_states[:, 2], linestyle='solid',
            color='olive', label='ground-truth')
    ax.plot(target_measurements[:, 0], target_measurements[:, 1], linestyle='None',
            marker='s', color='yellowgreen', label='target (measured)')
    ax.plot(clutters[:, 0], clutters[:, 1], marker='x', linestyle='None',
            color='orange', label='clutter')
    ax.plot(estimations[:, 0], estimations[:, 2], linestyle='dotted',
            marker='o', color='dodgerblue', alpha=0.6, label='estimate (track)')
    ax.plot(predictions[1:, 0], predictions[1:, 1], linestyle='None',
            marker='+', color='red', alpha=0.6, label='predict')

    set_label = False
    for i, elp_data in enumerate(valid_regions):
        if elp_data is None:
            continue

        label = 'validation gate' if not set_label else None
        set_label = True

        e = pat.Ellipse(xy=elp_data.center,
                        width=elp_data.width, height=elp_data.height,
                        angle=elp_data.angle,
                        ec="red", fc="none", alpha=0.6, label=label, zorder=5)
        ax.add_patch(e)

    ax.legend()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)

    fname = os.path.join(args.output_dir, f'trajectory')
    if args.seed >= 0:
        fname += f'_seed{args.seed}'
    plt.savefig(fname + '.png')

    fig = plt.figure(2, (8, 11))

    ax = fig.add_subplot(3, 1, 1)
    steps = cfg['steps']
    ax.plot(range(steps), target_states[:, 1]*3600/1000)
    ax.plot(range(steps), estimations[:, 1]*3600/1000)
    ax.legend(['truth', 'estimate'], loc='lower left')
    ax.set_ylabel('Velocity-x [km/h]')
    ax.grid(True)

    ax = fig.add_subplot(3, 1, 2)
    fig.subplots_adjust(bottom=0.1, top=0.95)
    steps = cfg['steps']
    ax.plot(range(steps), target_states[:, 3]*3600/1000)
    ax.plot(range(steps), estimations[:, 3]*3600/1000)
    ax.legend(['truth', 'estimate'], loc='lower left')
    ax.set_ylabel('Velocity-y [km/h]')
    ax.grid(True)

    is_imm = cfg['estimator']['IMM']
    ax = fig.add_subplot(3, 1, 3)
    steps = cfg['steps']
    if is_imm:
        ax.plot(range(steps), target_states[:, 4])
        ax.plot(range(steps), estimations[:, 4])
        ax.legend(['truth', 'estimate'], loc='lower left')
    else:
        ax.plot(range(steps), target_states[:, 4])
        ax.legend(['truth'], loc='lower left')
    ax.set_ylabel('Yaw-rate [rad/s]')
    ax.set_xlabel('steps')
    ax.grid(True)

    fname = os.path.join(args.output_dir, f'velocity')
    if args.seed >= 0:
        fname += f'_seed{args.seed}'
    plt.savefig(fname + '.png')


def extract_measurements(measurement_set_all):
    """Extract measured targets and clutters."""

    z_meas_hist = np.empty((0, 2))
    clutters = np.empty((0, 2))

    for t in range(len(measurement_set_all)):
        labels = measurement_set_all[t][0]
        measurements = measurement_set_all[t][1]

        for i in range(labels.shape[0]):
            meas = np.expand_dims(measurements[i], axis=0)
            if labels[i] == 1:
                z_meas_hist = np.append(z_meas_hist, meas, axis=0)
            elif labels[i] == 0:
                clutters = np.append(clutters, meas, axis=0)

    return z_meas_hist, clutters


def main():

    args = parse_arguments()

    if args.seed >= 0:
        np.random.seed(args.seed)

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    tracker = make_target_tracker(cfg)
    x_tgt_hist, z_tgt_hist, modes_tgt = generate_target_data(cfg)

    measurement_sets, x_posts, z_priors, valid_regions, mode_proba = \
        track_target(cfg, tracker, z_tgt_hist)

    z_meas_hist, clutters = extract_measurements(measurement_sets)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_trajectory(cfg, args, x_tgt_hist, z_meas_hist, clutters,
                    x_posts, z_priors, valid_regions)


if __name__ == '__main__':
    main()
