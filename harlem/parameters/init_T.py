import numpy as np
from harlem.utils import SQRT_DBL_EPSILON


def init_T(full_data, Qc, dGn, delta_star, x_grid, v_grid):

    w, dGn = np.array(sorted(zip(full_data.w, dGn))).T

    T1 = np.zeros((len(x_grid), len(v_grid)))
    T2 = np.zeros((len(x_grid), len(v_grid)))

    for i, x in enumerate(x_grid):
        for j, v in enumerate(v_grid):

            if v > x + delta_star:

                if v > x + delta_star:
                    mask_w = (x < w) & (w < v)
                    denominator = (dGn[mask_w] * Qc(v - w[mask_w])).sum()

                if denominator > SQRT_DBL_EPSILON:
                    T1[i, j] = (v - x) / denominator
                    T2[i, j] = 1 / denominator

    return T1, T2
