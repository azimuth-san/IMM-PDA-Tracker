from dataclasses import dataclass
import numpy as np
import numpy.linalg as la


class EllipseShape:

    def __init__(self, center, width, height, angle):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle

    @classmethod
    def from_quadratic_form(cls, A, value, center):
        """ Create EllpseShape from quadratic form

        quadtacit form : x.T @ A @ x = value

        lambda, U = eig(A)
        => x.T @ A @ x
           = x.T @ U @ lambda @ U.T @ x
           = y.T @ lambda @ y,  y = U.T @ x
           = [y1, ..., yn].T @ diag([lam_1, ... lam_n]) @ [y1, ..., yn]
           = lam_1 * (y1**2) + ... lam_n * (yn**2) = value
        """

        lambda_, U = la.eig(A)  # A = U @ np.diag(lambda_) @ U.T

        angle = np.arccos(np.dot(U[:, 0], np.array([1, 0])) / la.norm(U[:, 0]))
        angle = angle * 360 / (2 * np.pi)

        width = 2 * np.sqrt(value / lambda_[0])
        height = 2 * np.sqrt(value / lambda_[1])

        return cls(center=center, width=width, height=height, angle=angle)
