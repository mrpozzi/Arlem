from abc import abstractmethod, ABCMeta

from scipy.stats import beta
from functools import partial


class JointDistribution(object):

    __metaclass__ = ABCMeta

    def __init__(self,
                 margins=None,
                 param_margins=None,
                 *args,
                 **kwargs
                 ):

        if margins is None:
            margins = {'x': beta, 'z': beta}
        if param_margins is None:
            param_margins = {'z': {'a': 20, 'b': 5},
                             'x': {'a': 5, 'b': 7}}

        self.margins = margins
        self.param_margins = param_margins

        self.marginal_pdfs = [partial(margins['x'].pdf, **param_margins['x']),
                              partial(margins['z'].pdf, **param_margins['z'])]

        self.marginal_cdfs = [partial(margins['x'].cdf, **param_margins['x']),
                              partial(margins['z'].cdf, **param_margins['z'])]

        self.marginal_rvss = [lambda n: margins['x'].rvs(size=n, **param_margins['x']),
                              lambda n: margins['z'].rvs(size=n, **param_margins['z'])]

        self.marginal_ppfs = [partial(margins['x'].ppf, **param_margins['x']),
                              partial(margins['z'].ppf, **param_margins['z'])]

    @abstractmethod
    def cdf(self, xz, margin=None):
        pass

    @abstractmethod
    def pdf(self, xz, margin=None):
        pass

    @abstractmethod
    def rvs(self, n):
        pass


