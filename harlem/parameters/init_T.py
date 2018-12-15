import numpy as np
from harlem.utils import SQRT_DBL_EPSILON


def init_T(full_data, Qc, dGn, delta_star, x_grid, v_grid):

    w, dGn = np.array(sorted(zip(full_data.w, dGn))).T

    def _compute_denominator(x, v):
        if v > x + delta_star:
            mask_w = (x < w) & (w < v)
            denominator = (dGn[mask_w] * Qc(v - w[mask_w])).sum()
        else:
            denominator = 0.0
        return denominator

    denominator = np.vectorize(_compute_denominator)(x_grid[:, None], v_grid)

    T2 = 1 / denominator
    T2[denominator < SQRT_DBL_EPSILON] = 0.0

    T1 = T2 * np.array([[(v-x) if v > x + delta_star else 0.0 for v in v_grid] for x in x_grid])

    return T1, T2

