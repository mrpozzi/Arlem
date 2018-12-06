import numpy as np
from scipy import interpolate

from harlem.utils import SQRT_DBL_EPSILON, quick_simpson
from harlem.estimator.train.abc_step import ABCStep
from harlem.parameters.init_Q import compute_normalization


class StepConstant(ABCStep):

    def __repr__(self):
        return super(StepConstant, self).__repr__()

    def iterate(self, Q2):

        Q2 = self.tmle_step(Q2)

        #if (any(Q2 < 0))
        #stop("Negative Elements in Q2...\a")

        if self.verbose:
            print("D: {norm_grad}, |  log-Likelihood = {log_likelihood}".format(
                norm_grad=self.norm_grad,
                log_likelihood=self.log_likelihood))

        self.converged = (abs(self.norm_grad) < self.tol)
        return Q2

    def tmle_step(self, Q2):

        t1_slice = np.apply_along_axis(lambda f: quick_simpson(f, self.v_grid), 1, self.T1 * Q2)
        t2_slice = np.apply_along_axis(lambda f: quick_simpson(f, self.v_grid), 1, self.T2 * Q2)

        # Compute the Gradient and the boundaries for the optimization problem
        gradient = (np.apply_along_axis(lambda t1: [0 if (np.abs(d) < SQRT_DBL_EPSILON) else n / d
                                                    for n, d in zip(t1, t1_slice)], 0, self.T1) -
                    np.apply_along_axis(lambda t2: [0 if (np.abs(d) < SQRT_DBL_EPSILON) else n / d
                                                    for n, d in zip(t2, t2_slice)], 0, self.T2))
        grad_max = np.abs(gradient).max()

        # Evaluate the Gradient @ the observed data
        grad_obs = interpolate.interp2d(self.x_grid, self.v_grid, gradient.T, kind='linear')(self.x, self.v)

        # Solve the Optimization Problem and find the epsilon

        epsilon_opt, f_max = self.optimize_eps(grad_obs, grad_max)

        # Compute Q2 @ observed data + compute empirical Mean and Variance

        offset = 0.0
        interp_Q2 = interpolate.interp2d(self.x_grid, self.v_grid, Q2.T, kind='linear')
        for x, v in zip(self.x, self.v):
            q2_obs = interp_Q2(x, v)
            offset += np.log(q2_obs)

        offset /= self.n_full
        log_likelihood = offset + f_max

        emp_mean = gradient.mean()
        emp_var = gradient.var()

        norm_grad = emp_mean / np.sqrt(emp_var)

        # Update Q2
        Q2 = (1 + gradient * epsilon_opt) * Q2
        normalizing_constant = compute_normalization(Q2, self.x_grid, self.v_grid)

        if self.verbose:
            print("Renormalization Constant: {}".format(normalizing_constant))

        Q2 /= normalizing_constant

        self.norm_grad = norm_grad
        self.log_likelihood = log_likelihood
        self.epsilon_opt = epsilon_opt

        return Q2

    def optimize_eps(self, grad_obs, grad_max, n_steps=100):

        epsx_boundary = 1 / grad_max
        eps_opt = 0.0

        f_max = self.crit_fun0(grad_obs, grad_max, eps_opt)

        step_eps = 2.0 * epsx_boundary / (n_steps - 1)

        eps = -epsx_boundary
        for i in range(n_steps):

            f = self.crit_fun0(grad_obs, grad_max, eps)

            if f >= f_max:
                f_max = f
                eps_opt = eps

            eps = min(eps + step_eps, epsx_boundary)

        return eps_opt, f_max

    def crit_fun0(self, grad_obs, grad_max, eps):

        if abs(eps) * grad_max > 1:
            return -1.0

        value = (np.log(1.0 + eps * grad_obs)).mean()

        return value

