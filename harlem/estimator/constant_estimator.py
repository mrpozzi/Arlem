import scipy.integrate as integrate

from harlem.estimator.abc_estimator import ABCEstimator
from harlem.utils import quick_simpson


class ConstantEstimator(ABCEstimator):

    def psi(self, Q2):

        self._init_R(Q2)

        integral_R = quick_simpson(self.R * self.lambda_grid, self.x_grid)

        lambda_int = integrate.quad(self.lambda_fun, 0, self.tau)[0]
        theta = integral_R / lambda_int

        return theta



