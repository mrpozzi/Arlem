import numpy as np

from statsmodels.distributions.empirical_distribution import StepFunction
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from lifelines import KaplanMeierFitter


def get_survival_function(time, event, entry=None):

    kmf = KaplanMeierFitter()
    kmf.fit(durations=time, event_observed=(event == 1), entry=entry)

    return StepFunction(kmf.survival_function_.index,
                        kmf.survival_function_['KM_estimate'],
                        sorted=True, side='right')


def compute_normalization(Q2, x_grid, v_grid):
    normalizing_constant = 0.0

    for i, _ in enumerate(x_grid[1:]):
        for j, _ in enumerate(v_grid[1:]):
            phi00 = Q2[i, j]
            phi10 = Q2[i, j + 1]
            phi01 = Q2[i + 1, j]
            phi11 = Q2[i + 1, j + 1]

            normalizing_constant += ((x_grid[i + 1] - x_grid[i]) * (v_grid[j + 1] - v_grid[j]) *
                                     ((phi00 + phi01 + phi11) / 3. +
                                      (phi00 + phi10 + phi11) / 3.) / 2.)
    return normalizing_constant


def init_Q(full_data, x_grid, v_grid,
           delta_star, tau=100, verbose=False):

    obs_data = full_data[(full_data.delta0 * full_data.delta1 == 1)]
    obs_data = obs_data[obs_data.x != 0]
    # obs_data = obs_data[obs_data.x + delta_star <= obs_data.v]

    Qc = get_survival_function(time=full_data.v - full_data.w,
                               event=1 - full_data.delta1)

    Sn = get_survival_function(time=full_data.v,
                               event=full_data.delta1,
                               entry=full_data.w)

    dGn = 1 / Sn(full_data.w)
    dGn[Sn(full_data.w) == 0] = 0
    dGn[dGn / sum(dGn) > 100 / full_data.shape[0]] = 0
    dGn = dGn / sum(dGn)

    if Qc(delta_star) <= 0:
        raise Exception("Initialization NOT well defined: Qc(delta*)={}".format(Qc(delta_star)))

    # w = full_data.w

    Q2 = np.zeros((len(x_grid), len(v_grid)))

    kde = KDEMultivariate([obs_data.v.T - obs_data.x.T - delta_star, obs_data.v.T],
                          bw='cv_ml', var_type='cc')

    for i, x in enumerate(x_grid):
        for j, v in enumerate(v_grid):
            if (delta_star <= v <= tau and
                    0 <= x <= (v - delta_star)):
                Q2[i, j] = kde.pdf([v - x - delta_star, v])

    h_opt = kde.bw
    if verbose:
        print("h Opt {}".format(h_opt))

    normalizing_constant = compute_normalization(Q2, x_grid, v_grid)

    if verbose:
        print("Renormalization Constant: {}".format(normalizing_constant))

    Q2 /= normalizing_constant

    return {'Sn': Sn, 'dGn': dGn, 'Qc': Qc}, Q2





