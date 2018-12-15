import sys
import math
import numpy as np
import scipy.integrate as integrate

SQRT_DBL_EPSILON = math.sqrt(sys.float_info.epsilon)

def get_lambda_matrix(lambda_fun, tau):
    i0lambda = integrate.quad(lambda_fun, 0, tau)[0]
    i1lambda = integrate.quad(lambda t: t * lambda_fun(t), 0, tau)[0]
    i2lambda = integrate.quad(lambda t: (t ** 2) * lambda_fun(t), 0, tau)[0]
    return np.array([[i0lambda, i1lambda], [i1lambda, i2lambda]])
