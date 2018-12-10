from abc import abstractmethod, ABCMeta

import numpy as np


class ABCStep(object):
    __metaclass__ = ABCMeta

    def __init__(self, full_data,
                 T1, T2, Q1,
                 x_grid, v_grid,
                 delta_star, tau=100,
                 lambda_fun=lambda x: 1 * ((x >= 10) & (x <= 70)),
                 tol=None, verbose=False):

        obs_data = full_data[(full_data.delta0 * full_data.delta1 == 1)]
        obs_data = obs_data[obs_data.x != 0]
        # obsData <- obsData[which(obsData$x+deltaStar<=obsData$v),]

        self.w = full_data.w
        self.v = obs_data.v
        self.x = obs_data.x
        self.n_full = obs_data.shape[0]

        if tol is None:
            tol = 1/ np.sqrt(self.n_full)

        self.T1 = T1
        self.T2 = T2
        self.Q1 = Q1
        self.x_grid = x_grid
        self.v_grid = v_grid
        self.delta_star = delta_star
        self.tol = tol
        self.tau = tau
        self.verbose = verbose

        self.lambda_fun = lambda_fun
        self.lambda_grid = lambda_fun(x_grid)

        self.normalized_gradient = None
        self.log_likelihood = None
        self.epsilon_opt = None
        self.converged = False

    def __repr__(self):
        repr_str = """{name}(
                gradient={gradient},
                log_likelihood={log_likelihood},
                epsilon_opt={epsilon_opt},
                converged={converged})""".format(
            name=self.__class__,
            gradient=self.normalized_gradient,
            log_likelihood=self.log_likelihood,
            epsilon_opt=self.epsilon_opt,
            converged=self.converged,
        )
        return repr_str

    @abstractmethod
    def iterate(self, Q2):
        pass

    @abstractmethod
    def tmle_step(self, Q2):
        pass


