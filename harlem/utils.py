import sys
import math
import numpy as np

SQRT_DBL_EPSILON = math.sqrt(sys.float_info.epsilon)


# Simpson Integration
def quick_simpson(f, nodes):
    if len(f) != len(nodes):
        raise Exception("Number of nodes different from number of function Values!!!")

    ab = [min(nodes), max(nodes)]
    m = len(nodes)
    h = (ab[1] - ab[0]) / m

    mask_even = np.array(range(len(f))) % 2 == 0
    mask_odd = np.array(range(len(f))) % 2 == 1

    # Simpson Formula
    return (h / 3 * (f[0] + 2 * sum(f[mask_even])
                     + 4 * sum(f[mask_odd]) + f[len(f) - 1]))

