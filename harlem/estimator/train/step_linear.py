import numpy as np
from scipy import interpolate

from scipy.optimize import minimize
from scipy.optimize import Bounds


from harlem.estimator.train.abc_step import ABCStep
from harlem.parameters.init_Q import compute_normalization
from harlem.utils import SQRT_DBL_EPSILON


class StepLinear(ABCStep):

    def __repr__(self):
        return super(StepLinear, self).__repr__()

    def iterate(self, Q2, gradient):

        Q2 = self.tmle_step(Q2, gradient)

        if (Q2 < 0).any():
            raise Exception("Negative Elements in Q2.")

        if self.verbose:
            print("D: {norm_grad}, |  log-Likelihood = {log_likelihood}".format(
                norm_grad=self.normalized_gradient,
                log_likelihood=self.log_likelihood))
        print("Stopping: {0} > {1}".format(np.abs(self.normalized_gradient).max(), self.tol))
        self.converged = (np.abs(self.normalized_gradient).max() < self.tol)
        return Q2

    def tmle_step(self, Q2, gradient):

        grad_max = [np.abs(gradient).max(),
                    np.apply_along_axis(lambda g: np.abs(g) * self.x_grid, 0, gradient).max()]

        # Evaluate the Gradient @ the observed data
        grad_obs = interpolate.interp2d(self.x_grid, self.v_grid, gradient.T, kind='cubic')(self.x, self.v)

        # Solve the Optimization Problem and find the epsilon
        eps_fun = self.crit_fun1(grad_obs)

        epsilon_opt = minimize(eps_fun, x0=np.array([0.0, 0.0]), method='SLSQP',
                               constraints=[{'type': 'ineq', 'fun': lambda eps: 1-np.array(grad_max).dot(np.abs(eps))},
                                            {'type': 'ineq', 'fun': lambda eps: np.abs(eps-2*SQRT_DBL_EPSILON).sum()}],
                               bounds=Bounds([-1 / grad_max[0], -1 / grad_max[1]],
                                             [1 / grad_max[0], 1 / grad_max[1]])).x

        print("Epsilon: {}".format(epsilon_opt))
        f_max = -eps_fun(epsilon_opt)

        # Compute Q2 @ observed data + compute empirical Mean and Variance
        interpolated_Q2 = interpolate.interp2d(self.x_grid, self.v_grid, Q2.T, kind='cubic')
        offset = np.mean([np.log(interpolated_Q2(x, v)) for x, v in zip(self.x, self.v)])
        log_likelihood = offset + f_max

        if self.verbose:
            print("Log-likelihood: {}".format(log_likelihood))

        emp_mean = [gradient.mean(),
                    np.apply_along_axis(lambda g: np.abs(g) * self.x_grid, 0, gradient).mean()]
        emp_var = [gradient.var(),
                   np.apply_along_axis(lambda g: np.abs(g) * self.x_grid, 0, gradient).var()]
        norm_grad = emp_mean / np.sqrt(emp_var)

        # Update Q2
        Q2 *= (1 + np.apply_along_axis(lambda g: (epsilon_opt[0] + epsilon_opt[1] * self.x_grid) * g, 0, gradient))
        normalizing_constant = compute_normalization(Q2, self.x_grid, self.v_grid)

        if self.verbose:
            print("Renormalization Constant: {}".format(normalizing_constant))

        Q2 /= normalizing_constant

        self.normalized_gradient = norm_grad
        self.log_likelihood = log_likelihood
        self.epsilon_opt = epsilon_opt

        return Q2

    def crit_fun1(self, grad_obs):

        def opt_fun(eps):
            value = -np.mean([np.log(1 + (eps[0] + eps[1] * x) * g)
                              for x, g in zip(self.x, grad_obs)])
            return value

        return opt_fun

