import numpy as np
import scipy.integrate as integrate
from scipy import interpolate

from harlem.estimator.debias.harlem import HarlemABC
from harlem.utils import get_lambda_matrix


# TODO: Fix this
class HarlemOneStep(HarlemABC):

    def fit(self):

        gradient = self.get_gradient(self.Q2)
        gradient /= self.n_obs * self.n_full

        influence_curve = interpolate.interp2d(self.x_grid, self.v_grid, gradient.T, kind='cubic')

        if len(self.init_psi) == 2:

            lambda_grad = np.array([[self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)],
                                    [x * self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)]])

            lambda_matrix = get_lambda_matrix(self.lambda_fun, self.tau)

            print(np.linalg.solve(lambda_matrix, np.sum(lambda_grad, axis=1))[:, 0])
            theta = self.init_psi - np.linalg.solve(lambda_matrix, np.sum(lambda_grad, axis=1))[:, 0]

        else:

            lambda_grad = [self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)]
            inv_lambda = 1 / integrate.quad(self.lambda_fun, 0, self.tau)[0]
            theta = self.init_psi - inv_lambda * np.mean(lambda_grad)

        self.theta_hat = theta
        return theta
