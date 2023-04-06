import numpy as np


class Detector:
    def __init__(self, PD):
        """A classs to detect a target and clutters."""
        self.PD = PD  # detection probability

    def detect(self, target, clutter):
        """Detect the target and clutters."""

        measurements = np.empty((0, target.shape[0]))
        labels = np.empty(0, np.int32)

        if np.random.uniform(low=0.0, high=1.0) < self.PD:
            # the target is detected at a detection probability.
            measurements = np.append(measurements, target[np.newaxis, :],
                                     axis=0)
            labels = np.append(labels, 1)

        # make clutters.
        if clutter.shape[0] > 0:
            measurements = np.append(measurements, clutter, axis=0)
            labels = np.append(labels,
                               np.zeros(clutter.shape[0], dtype=int))

        # shuffle measurements randamoly.
        order = np.random.permutation(np.arange(labels.shape[0]))
        measurements = measurements[order, :]
        labels = labels[order]

        return measurements, labels
