import numpy as np
import scipy.integrate as integrate

from harlem.estimator.abc_estimator import ABCEstimator
from scipy.integrate import simps


class LinearEstimator(ABCEstimator):

    def psi(self, Q2):
        self._init_R(Q2)

        integral_R = [simps(self.R * self.lambda_grid, self.x_grid),
                      simps(self.R * self.lambda_grid * self.x_grid, self.x_grid)]

        i0lambda = integrate.quad(self.lambda_fun, 0, self.tau)[0]
        i1lambda = integrate.quad(lambda t: t * self.lambda_fun(t), 0, self.tau)[0]
        i2lambda = integrate.quad(lambda t: (t ** 2) * self.lambda_fun(t), 0, self.tau)[0]
        lambda_matrix = np.array([[i0lambda, i1lambda], [i1lambda, i2lambda]])

        return np.linalg.solve(lambda_matrix, integral_R)



