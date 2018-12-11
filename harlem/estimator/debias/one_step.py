import numpy as np
import scipy.integrate as integrate
from scipy import interpolate

from harlem.estimator.debias.harlem import HarlemABC
from harlem.utils import SQRT_DBL_EPSILON
from scipy.integrate import simps


class HarlemOneStep(HarlemABC):

    def _process_t(self, ti, t_slice):

        res = [1 if (abs(t) <= SQRT_DBL_EPSILON) and
                    (abs(s) <= SQRT_DBL_EPSILON) else t / s
               for t, s in zip(ti, t_slice)]
        res = [0 if lam == 0 else r for r, lam in zip(res, self.lambda_grid)]
        return res

    def fit(self):

        t1_slice = self.lambda_grid * np.apply_along_axis(lambda f: simps(f, self.v_grid), 1, self.T1 * self.Q2)
        t2_slice = self.lambda_grid * np.apply_along_axis(lambda f: simps(f, self.v_grid), 1, self.T2 * self.Q2)

        gradient = (np.apply_along_axis(lambda f: self._process_t(f, t1_slice), 0, self.T1) -
                    np.apply_along_axis(lambda f: self._process_t(f, t2_slice), 0, self.T2)) / (self.n_obs / self.n_full)
        gradient[self.Q2 == 0] = 0.0

        influence_curve = interpolate.interp2d(self.x_grid, self.v_grid, gradient, kind='cubic')

        if len(self.init_psi) == 2:

            lambda_grad = np.array([[self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)],
                                    [x * self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)]])

            i0lambda = integrate.quad(self.lambda_fun, 0, self.tau)[0]
            i1lambda = integrate.quad(lambda t: t * self.lambda_fun(t), 0, self.tau)[0]
            i2lambda = integrate.quad(lambda t: (t ** 2) * self.lambda_fun(t), 0, self.tau)[0]
            lambda_matrix = np.array([[i0lambda, i1lambda], [i1lambda, i2lambda]])

            return self.init_psi - np.linalg.solve(lambda_matrix, np.mean(lambda_grad, axis=1))[:, 0]

        else:

            lambda_grad = [self.lambda_fun(x) * influence_curve(x, v) for x, v in zip(self.x, self.v)]
            inv_lambda = 1 / integrate.quad(self.lambda_fun, 0, self.tau)[0]
            return self.init_psi - inv_lambda * np.mean(lambda_grad)
