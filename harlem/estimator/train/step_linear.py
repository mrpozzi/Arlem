import numpy as np
from scipy import interpolate
from scipy.integrate import simps

from scipy.optimize import minimize
from scipy.optimize import Bounds

from harlem.utils import SQRT_DBL_EPSILON
from harlem.estimator.train.abc_step import ABCStep
from harlem.parameters.init_Q import compute_normalization


class StepLinear(ABCStep):

    def __repr__(self):
        return super(StepLinear, self).__repr__()

    def iterate(self, Q2):

        Q2 = self.tmle_step(Q2)

        if (Q2 < 0).any():
            raise Exception("Negative Elements in Q2.")

        if self.verbose:
            print("D: {norm_grad}, |  log-Likelihood = {log_likelihood}".format(
                norm_grad=self.normalized_gradient,
                log_likelihood=self.log_likelihood))
        print("Stopping: {0} - {1}".format(np.abs(self.normalized_gradient).max(), self.tol))
        self.converged = (np.abs(self.normalized_gradient).max() < self.tol)
        return Q2

    def tmle_step(self, Q2):

        t1_slice = np.apply_along_axis(lambda f: simps(f, self.v_grid), 0, self.T1 * Q2)
        t2_slice = np.apply_along_axis(lambda f: simps(f, self.v_grid), 0, self.T2 * Q2)

        # Compute the Gradient and the boundaries for the optimization problem
        gradient = (np.apply_along_axis(lambda t1: [0 if (np.abs(d) < SQRT_DBL_EPSILON) else n / d
                                                    for n, d in zip(t1, t1_slice)], 1, self.T1) -
                    np.apply_along_axis(lambda t2: [0 if (np.abs(d) < SQRT_DBL_EPSILON) else n / d
                                                    for n, d in zip(t2, t2_slice)], 1, self.T2))

        grad_max = [np.abs(gradient).max(),
                    np.apply_along_axis(lambda g: np.abs(g) * self.x_grid, 1, gradient).max()]

        # Evaluate the Gradient @ the observed data
        grad_obs = interpolate.interp2d(self.x_grid, self.v_grid, gradient, kind='cubic')(self.x, self.v)

        # Solve the Optimization Problem and find the epsilon
        eps_fun = self.crit_fun1(grad_obs, grad_max)
        epsilon_opt = minimize(eps_fun, x0=np.array([0.01, 0.01]), method='L-BFGS-B',
                               bounds=Bounds([-1 / grad_max[0], -1 / grad_max[1]],
                                             [1 / grad_max[0], 1 / grad_max[1]])).x
        f_max = -eps_fun(epsilon_opt)

        # Compute Q2 @ observed data + compute empirical Mean and Variance

        interp_Q2 = interpolate.interp2d(self.x_grid, self.v_grid, Q2.T, kind='cubic')
        offset = sum([np.log(interp_Q2(x, v)) for x, v in zip(self.x, self.v)]) / self.n_full
        log_likelihood = offset + f_max
        if self.verbose:
            print("Log-likelihood: {}".format(log_likelihood))

        emp_mean = [gradient.mean(),
                    np.apply_along_axis(lambda g: np.abs(g) * self.x_grid, 1, gradient).mean()]
        emp_var = [gradient.var(),
                   np.apply_along_axis(lambda g: np.abs(g) * self.x_grid, 1, gradient).var()]

        norm_grad = [emp_mean[0] / np.sqrt(emp_var[0]),
                     emp_mean[1] / np.sqrt(emp_var[1])]

        # Update Q2
        print(np.apply_along_axis(lambda g: (epsilon_opt[0] +
                                                 epsilon_opt[1] * self.x_grid) * g,
                                      1, gradient).min())
        # Q2 = (1 + (epsilon_opt[0] + epsilon_opt[1] * self.x_grid) * gradient) * Q2

        Q2 = (1 + np.apply_along_axis(lambda g: (epsilon_opt[0] +
                                                 epsilon_opt[1] * self.x_grid) * g,
                                      1, gradient)) * Q2
        normalizing_constant = compute_normalization(Q2, self.x_grid, self.v_grid)

        if self.verbose:
            print("Renormalization Constant: {}".format(normalizing_constant))

        Q2 /= normalizing_constant

        self.normalized_gradient = norm_grad
        self.log_likelihood = log_likelihood
        self.epsilon_opt = epsilon_opt

        return Q2

    def crit_fun1(self, grad_obs, grad_max):

        def opt_fun(eps):
            if (np.abs(eps[0]) * grad_max[0] + np.abs(eps[1]) * grad_max[1]) > 1:
                return 1.0
            value = -np.mean([np.log(1 + (eps[0] + eps[1] * x) * g) for x, g in zip(self.x, grad_obs)])

            return value

        return opt_fun

