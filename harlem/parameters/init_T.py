import numpy as np
from harlem.utils import SQRT_DBL_EPSILON


def init_T(full_data, Qc, dGn, delta_star, x_grid, v_grid):

    w, dGn = zip(*sorted(zip(full_data.w, dGn)))

    T1 = np.zeros((len(x_grid), len(v_grid)))
    T2 = np.zeros((len(x_grid), len(v_grid)))

    for i, x in enumerate(x_grid):
        print("{} ".format(i/len(x_grid)), end="", flush=True)
        for j, v in enumerate(v_grid):

            if v > x + delta_star:
                denominator = 0.0
                k = 0

                while (k < len(w)) and (w[k] < v):#(x < w[k] < v):
                    if (x < w[k]):
                        denominator += dGn[k] * Qc(v - w[k])
                    k += 1

                if denominator > SQRT_DBL_EPSILON:
                    T1[i, j] = (v - x) / denominator
                    T2[i, j] = 1 / denominator

    return T1, T2
