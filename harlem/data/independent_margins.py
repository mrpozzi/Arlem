from harlem.data.joint_distribution import JointDistribution
import numpy as np


class IndependentMargins(JointDistribution):

    def __init__(self,
                 margins=None,
                 param_margins=None,
                 *args,
                 **kwargs
                 ):
        super(IndependentMargins, self).__init__(margins, param_margins, *args, **kwargs)

    def cdf(self, xz, margin=None):
        if not margin:
            log_cdf = sum([np.log(cdf(xz[i])) for i, cdf in enumerate(self.marginal_cdfs)])
            return np.exp(log_cdf)
        else:
            return self.marginal_cdfs[margin](xz)

    def pdf(self, xz, margin=None):

        if not margin:
            log_pdf = sum([np.log(pdf(xz[i])) for i, pdf in enumerate(self.marginal_pdfs)])
            return np.exp(log_pdf)
        else:
            return self.marginal_pdfs[margin](xz)

    def rvs(self, n):
        return [rvs(n) for rvs in self.marginal_rvss]
