import numpy as np

from statsmodels.sandbox.distributions.copula import CopulaBivariate
from functools import partial
from scipy.special import expm1
from scipy.stats import uniform

from harlem.data.joint_distribution import JointDistribution


class FrankCopula(object):

    def __init__(self, theta):
        self.theta = theta

    def copula_bv(self, u, v, *args):
        """
        Cook, Johnson bivariate copula with negative correlation
        """
        cdfv = -np.log(1 + expm1(-self.theta * u) *
                       expm1(-self.theta * v) / expm1(-self.theta)) / self.theta
        cdfv = np.minimum(cdfv, 1)
        return cdfv

    def _g(self, z):
        """Helper function to solve Frank copula.
        This functions encapsulates :math:`g_z = e^{-\\theta z} - 1` used on Frank copulas.
        Argument:
            z:
        Returns:

        """
        return np.exp(np.multiply(-self.theta, z)) - 1

    def copula_bv_density(self, u, v):
        """Compute density function for given copula family.
        Args:
            u:
            v:
        Returns:
            : probability density
        """
        if self.theta == 0:
            return np.multiply(u, v)

        else:
            num = np.multiply(np.multiply(-self.theta, self._g(1)), 1 + self._g(np.add(u, v)))
            aux = np.multiply(self._g(u), self._g(v)) + self._g(1)
            den = np.power(aux, 2)
            return num / den

    def _psi_inv(self, u, p):
        return -np.log((expm1(-self.theta) + expm1(-self.theta * u) * (1-p)/p) /
                       (1 + expm1(-self.theta * u) * (1-p)/p)) / self.theta


class HarlemCopula(JointDistribution):

    def __init__(self,
                 margins=None,
                 param_margins=None,
                 copula=FrankCopula(-5),
                 copula_args=None,
                 *args,
                 **kwargs
                 ):

        super(HarlemCopula, self).__init__(margins, param_margins, *args, **kwargs)
        self.copula = copula

        self.copula_dist = CopulaBivariate(marginalcdfs=[partial(self.margins['x'].cdf, **self.param_margins['x']),
                                                         partial(self.margins['z'].cdf, **self.param_margins['z'])],
                                           copula=self.copula.copula_bv,
                                           copargs=copula_args or [])

    def cdf(self, xz, margin=None):
        if not margin:
            return self.copula_dist.cdf(xz)
        else:
            return self.marginal_cdfs[margin](xz)

    def pdf(self, xz, margin=None):
        if not margin:

            transformed = [cdf(x) for cdf, x in zip(self.marginal_cdfs, xz)]

            base_cdf = self.copula.copula_bv_density(transformed[0], transformed[1])
            log_pdf = sum([np.log(pdf(xz[i])) for i, pdf in enumerate(self.marginal_pdfs)])

            return base_cdf * np.exp(log_pdf)
        else:
            return self.marginal_pdfs[margin](xz)

    def rvs(self, n):
        u = uniform.rvs(size=n)
        p = uniform.rvs(size=n)
        v = self.copula._psi_inv(u, p)
        return np.array([u, v])

