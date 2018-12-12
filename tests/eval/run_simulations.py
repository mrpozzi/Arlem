import json
import numpy as np

from harlem.data.data_generator import DataGeneratorHarlem
from harlem.estimator.linear_estimator import LinearEstimator
from harlem.estimator.debias.one_step import HarlemOneStep
from harlem.estimator.debias.tmle import HarlemTMLE
from harlem.parameters.init_Q import init_Q
from harlem.parameters.init_T import init_T

SAMPLE_SIZES = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
NUM_SIMS = 100

def run_scenario(data_gen, sample_size, tau=100):

    n_grid = 300
    if sample_size > 10000:
        n_grid = 500
    elif sample_size == 20000:
        n_grid = 600

    full_data, seed = data_gen.data_generator(sample_size)

    delta_star = max(np.abs(np.diff(sorted(full_data.w))))
    print(delta_star)

    x_grid = np.linspace(0, tau, num=n_grid)
    v_grid = np.linspace(0, tau, num=n_grid)

    Q0 = init_Q(full_data, x_grid, v_grid, delta_star, tau=tau, verbose=True)
    T0 = init_T(full_data, Q0[0]['Qc'], Q0[0]['dGn'], delta_star, x_grid, v_grid)

    linear_estimator = LinearEstimator(full_data, T0[0], T0[1], Q0[0],
                                       x_grid, v_grid, delta_star)
    theta_linear = linear_estimator.psi(Q0[1])

    one_step = HarlemOneStep(full_data=full_data, delta_star=delta_star, T0=T0, Q0=Q0)
    one_step_theta = one_step.fit()

    tmle = HarlemTMLE(full_data=full_data, delta_star=delta_star, Q0=Q0, T0=T0, verbose=True)
    tmle_theta = tmle.fit()

    results = {"seed": seed, "init_psi": theta_linear, "tmle": tmle_theta, "one_step": one_step_theta}
    return results

def run_simulations(data_gen, sample_size):

    simulation_results = []
    for i in range(NUM_SIMS):
        print("{} ".format(i / NUM_SIMS), end="", flush=True)
        simulation_results.append(run_scenario(data_gen, sample_size))
    print("\n")
    return simulation_results



if __name__== "__main__":
    distribution = DataGeneratorHarlem()

    true_theta = distribution.theta
    all_simulation_results = {}
    for sample_size in SAMPLE_SIZES:
        print("Sample Size: {}".format(sample_size))
        all_simulation_results[sample_size] = run_simulations(distribution, sample_size)
        with open('sample_size_{}_{}.json'.format(sample_size, NUM_SIMS), 'w') as outfile:
            json.dump(all_simulation_results[sample_size], outfile)

    all_simulation_results["true_theta"] = true_theta

    with open('harlem_simulations_{}.json'.format(NUM_SIMS), 'w') as outfile:
        json.dump(all_simulation_results, outfile)
