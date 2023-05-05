import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.linalg import solve
from scipy.optimize import minimize

# Define the system of differential equations
def dpop_dt(pop, t, params):
    P1, P2, Q1, Q2 = pop
    alpha1, alpha2, beta11, beta12, beta21, beta22, delta11, delta12, delta21, delta22, gamma1, gamma2, psi12, psi21, K1, K2, xi12, xi21, L1, L2 = params
    
    dP1_dt = alpha1 * P1 * (1 - (P1 + psi12 * P2) / K1) - beta11 * P1 * Q1 - beta12 * P1 * Q2
    dP2_dt = alpha2 * P2 * (1 - (P2 + psi21 * P1) / K2) - beta21 * P2 * Q1 - beta22 * P2 * Q2
    dQ1_dt = delta11 * P1 * Q1 + delta21 * P2 * Q1 - gamma1 * Q1 * (1 - (Q1 + xi12 * Q2) / L1)
    dQ2_dt = delta12 * P1 * Q2 + delta22 * P2 * Q2 - gamma2 * Q2 * (1 - (Q2 + xi21 * Q1) / L2)
    return [dP1_dt, dP2_dt, dQ1_dt, dQ2_dt]

# Set initial populations and parameters
initial_populations = [0.25, 3.66328193, 2.57955906, 0.251883]
params = [1., 1., 0.3, 0.5, 0.3, 0.4, 0.1, 0.5, 0.2, 0.1, 0.8, 0.5, 0.3, 0.5, 100, 100, 0.2, 1, 50, 50]
t = np.arange(0,5000,0.0005)

# Solve the system of equations
populations = odeint(dpop_dt, initial_populations, t, args=(params,), )

# Animate the results
# fig, ax = plt.subplots()

def update(num):
    ax.clear()
    ax.plot(populations[:num, 0], populations[:num, 1], label='Prey 1', color='blue')
    ax.plot(populations[:num, 2], populations[:num, 3], label='Predator 1', color='red')
    ax.plot(populations[:num, 0], populations[:num, 2], label='Prey 2', color='green')
    ax.plot(populations[:num, 1], populations[:num, 3], label='Predator 2', color='purple')
    ax.legend(loc='best')
    ax.set_xlabel('Prey Species')
    ax.set_ylabel('Predator Species')

# ani = FuncAnimation(fig, update, frames=range(len(t)), repeat=False, interval=0)
# plt.show()

def steady(params):
    alpha1, alpha2, beta11, beta12, beta21, beta22, delta11, delta12, delta21, delta22, gamma1, gamma2, psi12, psi21, K1, K2, xi12, xi21, L1, L2 = params
    A = np.array([
        [-alpha1/K1, -alpha1/K1*psi12, -beta11, -beta12],
        [-alpha2/K2*psi21, -alpha2/K2, -beta21, -beta22],
        [delta11, delta21, gamma1/L1, gamma1/L1*xi12],
        [delta12, delta22, gamma2/L2*xi21, gamma2/L2],
    ])
    b = [-alpha1, -alpha2, gamma1, gamma2]
    x = solve(A,b)
    print(x)
    return x

steady(params)

import math
len = math.floor(len(t)/10)
t_last = t[-len:]

fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Time series plot for prey species
axs[0].plot(t_last, populations[:, 0][-len:], label='Prey 1', color='blue')
axs[0].plot(t_last, populations[:, 1][-len:], label='Prey 2', color='green')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Population')
axs[0].set_title('Time Series Plot for Prey Species')
axs[0].legend(loc='best')

# Time series plot for predator species
axs[1].plot(t_last, populations[:, 2][-len:], label='Predator 1', color='red')
axs[1].plot(t_last, populations[:, 3][-len:], label='Predator 2', color='purple')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Population')
axs[1].set_title('Time Series Plot for Predator Species')
axs[1].legend(loc='best')

plt.tight_layout()
plt.show()
