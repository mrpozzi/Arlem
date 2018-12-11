import numpy as np
import pandas as pd
import random
import time

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.stats import beta, expon

from harlem.parameters.init_T import SQRT_DBL_EPSILON
from harlem.data.harlem_copula import HarlemCopula


class DataGeneratorHarlem(object):

    def __init__(self,
                 delta=3,
                 tau=100,
                 lambda_fun=lambda x: 1 * ((x >= 10) & (x <= 70)),
                 par=None,
                 rate=0.017,
                 joint_distribution=HarlemCopula(),
                 n_grid=200):

        if not par:
            par = [1, 1]

        self.delta = delta
        self.par = par
        self.tau = tau
        self.lambda_fun = lambda_fun
        self.rate = rate
        self.joint_distribution = joint_distribution

        self.theta = self._get_theta(n_grid)

    def _get_R(self, grid_x):

        Sz = lambda u: 1 - self.joint_distribution.cdf(u / self.tau, 1)
        intSz = lambda t: integrate.quad(Sz, t, self.tau)[0]

        numerator = np.array([intSz(x) for x in grid_x])
        denominator = np.array([Sz(x) for x in grid_x])

        R1 = [1 if (n < SQRT_DBL_EPSILON) & (d < SQRT_DBL_EPSILON) else n / d
              for n, d in zip(numerator, denominator)]

        def R2_fun(x):

            def slice1(z):
                p = (z - x) * self.joint_distribution.pdf([x / self.tau, z / self.tau])
                if p is np.nan:
                    return 0.
                return p

            def slice2(z):
                p = self.joint_distribution.pdf([x / self.tau, z / self.tau])
                if p is np.nan:
                    return 0.
                return p

            num = integrate.quad(slice1, x + self.delta, self.tau)[0]
            den = integrate.quad(slice2, x + self.delta, self.tau)[0]

            res = 1 if (num < SQRT_DBL_EPSILON) & (den < SQRT_DBL_EPSILON) else num / den

            return res
        R2 = [R2_fun(x) for x in grid_x]

        R = np.log(R1) - np.log(R2)
        R[np.isinf(R)] = 0
        return R

    def _get_theta(self, n_grid):

        grid_x = np.linspace(0, self.tau, n_grid)
        lambda_grid = self.lambda_fun(grid_x)

        R = self._get_R(grid_x)

        def R_fun_gen ():
            support = grid_x[lambda_grid != 0]
            r_fit = interp1d(support, R[lambda_grid != 0], kind='cubic')
            return lambda t: 0 if self.lambda_fun(t) == 0 or not (min(support) < t < max(support)) else r_fit(t)

        R_fun = R_fun_gen()
        integral_R = [integrate.quad(lambda t: self.lambda_fun(t) * R_fun(t), 0, self.tau)[0],
                      integrate.quad(lambda t: self.lambda_fun(t) * t * R_fun(t), 0, self.tau)[0]]

        i0lambda = integrate.quad(self.lambda_fun, 0, self.tau)[0]
        i1lambda = integrate.quad(lambda t: t * self.lambda_fun(t), 0, self.tau)[0]
        i2lambda = integrate.quad(lambda t:  (t ** 2) * self.lambda_fun(t), 0, self.tau)[0]
        lambda_matrix = np.array([[i0lambda, i1lambda], [i1lambda, i2lambda]])

        return np.linalg.solve(lambda_matrix, integral_R)

    def data_generator(self, n, seed=int(time.time())):

        random.seed(seed)

        print("seed: {}".format(seed))

        full_data = pd.DataFrame(columns=["w", "x", "v", "delta0", "delta1"])
        count = 0

        while count < n:

            z, x = self.joint_distribution.rvs(n-count)
            w = beta.rvs(a=self.par[0], b=self.par[1], size=n-count)

            is_valid = (z > w)
            n_valid = sum(is_valid)

            if n_valid != 0:

                c = expon.rvs(loc=1/self.rate, size=n_valid)

                w = self.tau * w[is_valid]
                x = self.tau * x[is_valid]
                z = self.tau * z[is_valid]
                delta0 = 1 * (z > x + self.delta)

                delta0[x >= w] = 0
                x *= delta0

                delta1 = 1 * (z <= w + c)

                new_data = pd.DataFrame({"x": x, "z": z, "w": w,  "delta0": delta0, "delta1": delta1, "c": c})

                new_data["v"] = new_data.apply(lambda row: min(self.tau, row["z"], row["w"] + row["c"]), axis=1)
                full_data = full_data.append(new_data, ignore_index=True, sort=True)

                count += n_valid

        full_data.drop(columns=["c", "z"], inplace=True)
        full_data.delta0 = full_data.delta0.astype('int64')
        full_data.delta1 = full_data.delta1.astype('int64')

        return full_data.reset_index(drop=True), seed




