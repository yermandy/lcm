import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

import numpy as np
import matplotlib.pyplot as plt

g_poly = np.array([-3, 0.02, 0.0001])
z1 = np.array([g_poly, -g_poly])

g_poly = np.array([-0.7, -0.04, -0.0001])
z2 = np.array([g_poly, -g_poly])

g_poly = np.array([-1, -0.02, 0.0002])
z3 = np.array([g_poly, -g_poly])

ages = np.arange(1, 91)
poly_deg_2 = np.array([ages ** d for d in range(3)])

sigmoid = lambda x: 1 / (1 + np.exp(-x))

fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 3 * 1))

for z, ax in zip([z1, z2, z3], axes):

    gamma = z @ poly_deg_2
    PgIag = sigmoid(-1 * gamma)

    ax.plot(ages, PgIag[0], label='female')
    ax.plot(ages, PgIag[1], label='male')

    ax.legend()

plt.show()