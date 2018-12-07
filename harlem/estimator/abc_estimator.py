from abc import abstractmethod, ABCMeta
import numpy as np
import scipy.integrate as integrate


from harlem.utils import SQRT_DBL_EPSILON, quick_simpson


class ABCEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self, full_data,
                 T1, T2, Q1,
                 x_grid, v_grid,
                 delta_star, tau=100,
                 lambda_fun=lambda x: 1 * ((x >= 10) & (x <= 70))):

        obs_data = full_data[(full_data.delta0 * full_data.delta1 == 1)]
        obs_data = obs_data[obs_data.x != 0]
        # obsData <- obsData[which(obsData$x+deltaStar<=obsData$v),]

        self.w = full_data.w
        self.v = obs_data.v

        self.v_grid = v_grid
        self.x_grid = x_grid
        self.delta_star = delta_star
        self.tau = tau

        self.T1 = T1
        self.T2 = T2

        self.lambda_fun = lambda_fun
        self.lambda_grid = lambda_fun(self.x_grid)

        self.R = None

        Sn = Q1['Sn']
        integral_Sn = lambda t: integrate.quad(Sn, t, self.tau)[0]

        numerator = np.array([integral_Sn(x) for x in x_grid])
        denominator = np.array([Sn(x) for x in x_grid])

        self.R1 = [1 if (n < SQRT_DBL_EPSILON) & (d < SQRT_DBL_EPSILON) else n / d
                   for n, d in zip(numerator, denominator)]

    def _init_R(self, Q2):

        t1_slice = np.apply_along_axis(lambda f: quick_simpson(f, self.v_grid), 1, self.T1 * Q2)
        t2_slice = np.apply_along_axis(lambda f: quick_simpson(f, self.v_grid), 1, self.T2 * Q2)

        R2 = [1 if (np.abs(d) < SQRT_DBL_EPSILON) else n / d
              for n, d in zip(t1_slice, t2_slice)]

        self.R = np.log(self.R1) - np.log(R2)
        self.R[np.isinf(self.R)] = 0

    @abstractmethod
    def psi(self, Q2):
        pass

