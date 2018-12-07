import pandas as pd
import numpy as np

from harlem.parameters.init_Q import init_Q
from harlem.parameters.init_T import init_T
from harlem.estimator.train.step_linear import StepLinear
from harlem.estimator.linear_estimator import LinearEstimator

from harlem.estimator.debias.harlem import HarlemABC


class HarlemTMLE(HarlemABC):

    def __init__(self,
                 full_data,
                 target_step=StepLinear,
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

        super(HarlemTMLE, self).__init__(full_data,  estimator, delta_star,
                                         tau, n_grid, lambda_fun, Q0, T0, verbose,
                                         *args, **kwargs)
        self.target_step = target_step(full_data, self.T1, self.T2, self.Q1,
                                       self.x_grid, self.v_grid,
                                       self.delta_star, self.tau,
                                       self.lambda_fun, verbose=verbose)
        self.iterations = 0
        self.has_converged = False

    def fit(self):

        # self.normalized_gradient = self.log_likelihood = []

        while not self.has_converged:

            self.iterations += 1

            self.Q2 = self.target_step.iterate(self.Q2)
            #if self.verbose:
            #    print("\n")
            self.has_converged = self.target_step.converged

            self.log_likelihood = self.target_step.log_likelihood
            self.normalized_gradient = self.target_step.normalized_gradient

        #print("\n")
        theta = self.estimator.psi(self.Q2)
        self.theta_hat = pd.concat(self.init_psi, theta)
