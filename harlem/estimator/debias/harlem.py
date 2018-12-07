from abc import abstractmethod, ABCMeta
import numpy as np

from harlem.parameters.init_Q import init_Q
from harlem.parameters.init_T import init_T
from harlem.estimator.linear_estimator import LinearEstimator


class HarlemABC(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 full_data,
                 estimator=LinearEstimator,
                 delta_star=None,
                 tau=100,
                 n_grid=100,
                 lambda_fun=lambda x: 1 * ((x >= 10) & (x <= 70)),
                 Q0=None,
                 T0=None,
                 verbose=False,
                 *args,
                 **kwargs):

        self.thetaHat = None
        self.lambda_fun = lambda_fun
        self.verbose = verbose

        self.log_likelihood = None
        self.normalized_gradient = None
        self.iter = 0

        self.tau = tau

        if n_grid % 2 == 1:
            n_grid += 1

        self.x_grid = np.linspace(0, self.tau, num=n_grid)
        self.v_grid = np.linspace(0, self.tau, num=n_grid)

        if not delta_star:
            delta_star = max(np.diff(sorted(full_data.w)))
        self.delta_star = delta_star

        obs_data = full_data[(full_data.delta0 * full_data.delta1 == 1)]
        obs_data = obs_data[obs_data.x != 0]
        self.n_obs = obs_data.shape[0]
        self.n_full = full_data.shape[0]
        self.x = obs_data.x
        self.v = obs_data.v

        if Q0 is None:
            if verbose:
                print("Initializing the Q's...\n")

            Q0 = init_Q(full_data, self.x_grid, self.v_grid,
                        delta_star=delta_star, tau=self.tau, verbose=verbose)

        self.Q1, self.Q2 = Q0
        self.Sn = self.Q1['Sn']
        self.Qc = self.Q1['Qc']
        self.dGn = self.Q1['dGn']

        if T0 is None:
            if verbose:
                print("Initializing the T's...\n")
            T0 = init_T(full_data, self.Qc, self.dGn,
                        delta_star=delta_star, x_grid=self.x_grid, v_grid=self.v_grid)

        self.T1, self.T2 = T0

        self.estimator = estimator(full_data, self.T1, self.T2, self.Q1,
                                   self.x_grid, self.v_grid, self.delta_star)

        self.lambda_grid = lambda_fun(self.x_grid)
        self.lambda_obs = lambda_fun(obs_data.x)

        self.init_psi = self.estimator.psi(self.Q2)
        self.theta_hat = None

    @abstractmethod
    def fit(self):
        pass
