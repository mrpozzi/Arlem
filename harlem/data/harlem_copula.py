import numpy as np

from statsmodels.sandbox.distributions.copula import CopulaBivariate
from functools import partial
from scipy.special import expm1

from harlem.data.joint_distribution import JointDistribution


def copula_bv_frank_neg(u, v, theta):
    '''
    Cook, Johnson bivariate copula with negative correlation
    '''
    cdfv = -np.log(1 + expm1(-theta*u) * expm1(-theta*v) / expm1(-theta))/theta
    cdfv = np.minimum(cdfv, 1)  #necessary for example if theta=100
    return cdfv


class HarlemCopula(JointDistribution):

    def __init__(self,
                 margins=None,
                 param_margins=None,
                 copula=copula_bv_frank_neg,
                 copula_args=None,
                 *args,
                 **kwargs
                 ):

        super(HarlemCopula).__init__(margins, param_margins, *args, **kwargs)

        self.copula_dist = CopulaBivariate(marginalcdfs=[partial(self.margins['x'].cdf, **self.param_margins['x']),
                                                         partial(self.margins['z'].cdf, **self.param_margins['z'])],
                                           copula=copula,
                                           copargs=copula_args or [-5])
        self.margin_map = {0: 'x', 1: 'z'}

    def cdf(self, xz, margin=None):
        if not margin:
            return self.copula_dist.cdf(xz)
        else:
            return self.copula_dist.marginalcdfs[self.margin_map[margin]].cdf(xz)

    def pdf(self, xz, margin=None):
        # return self.copula_dist.cdf(xz)
        pass

    def rvs(self, n):
        pass

