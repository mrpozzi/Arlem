import numpy as np
import scipy.integrate as integrate
from scipy import interpolate

from harlem.estimator.debias.harlem import HarlemABC


# TODO: Fix this
class HarlemOneStep(HarlemABC):

    def fit(self):

        gradient = self.get_gradient(self.Q2)
        gradient /= self.n_obs * self.n_full

        influence_curve = interpolate.interp2d(self.x_grid, self.v_grid, gradient.T, kind='cubic')

        if len(self.init_psi) == 2:

            lambda_grad = np.array([[self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)],
                                    [x * self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)]])

            i0lambda = integrate.quad(self.lambda_fun, 0, self.tau)[0]
            i1lambda = integrate.quad(lambda t: t * self.lambda_fun(t), 0, self.tau)[0]
            i2lambda = integrate.quad(lambda t: (t ** 2) * self.lambda_fun(t), 0, self.tau)[0]
            lambda_matrix = np.array([[i0lambda, i1lambda], [i1lambda, i2lambda]])

            print(np.linalg.solve(lambda_matrix, np.sum(lambda_grad, axis=1))[:, 0])
            theta = self.init_psi - np.linalg.solve(lambda_matrix, np.sum(lambda_grad, axis=1))[:, 0]

        else:

            lambda_grad = [self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)]
            inv_lambda = 1 / integrate.quad(self.lambda_fun, 0, self.tau)[0]
            theta = self.init_psi - inv_lambda * np.mean(lambda_grad)

        self.theta_hat = theta
        return theta
