import numpy as np
from harlem.data.data_generator import DataGeneratorHarlem
from harlem.parameters.init_Q import init_Q, initT

#cat("SIM ", i, "\n")



tau = 100

def lambda_const(tau):

    return lambda x: (x >= 10) * (x <= 70)



data_gen = DataGeneratorHarlem()
truth = data_gen.theta

n_grid=100
x_grid = np.linspace(0, tau, num=n_grid)
v_grid = np.linspace(0, tau, num=n_grid)

full_data, _ = data_gen.data_generator(1000)



#seed < -attr(fullData, "seed")

obs_data = full_data[(full_data.delta0 * full_data.delta1 == 1)]
#obs_data = obs_data[obs_data.x != 0]

delta_star = max(np.diff(sorted(full_data.w)))

print("Initializing the Q's...")
Q1, Q2 = init_Q(full_data, x_grid, v_grid, delta_star, tau=tau, verbose=True)
#hOpt = attr(Q2, "hOpt")

print("Initializing the T's...")
T1, T2 = initT(full_data, Q1['Qc'], Q1['dGn'], delta_star, x_grid, v_grid)

print(("Initializing  Psi..."))
#initPsi < - linearPsi(fullData, deltaStar, tau)(T1, T2, Q1, Q2, lambda_const, xGrid, vGrid)

print("Running TMLE...")
#TMLE < - harlemTMLE(fullData, stepLinear(fullData, deltaStar, tau, lambda_const, 1 / sqrt(nrow(obsData))),
#                    linearPsi(fullData, deltaStar, tau), deltaStar, tau, nGrid=nGrid, Q0=Q0, T0=T0)
