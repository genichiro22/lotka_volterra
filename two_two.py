import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
initial_populations = [40, 50, 10, 20]
params = [1.2, 1.1, 0.3, 0.4, 0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.3, 0.3, 100, 100, 0.2, 0.2, 50, 50]
t = np.linspace(0, 100, 1000)

# Solve the system of equations
populations = odeint(dpop_dt, initial_populations, t, args=(params,))

# Animate the results
fig, ax = plt.subplots()

def update(num):
    ax.clear()
    ax.plot(populations[:num, 0], populations[:num, 1], label='Prey 1', color='blue')
    ax.plot(populations[:num, 2], populations[:num, 3], label='Predator 1', color='red')
    ax.plot(populations[:num, 0], populations[:num, 2], label='Prey 2', color='green')
    ax.plot(populations[:num, 1], populations[:num, 3], label='Predator 2', color='purple')
    ax.legend(loc='best')
    ax.set_xlabel('Prey Species')
    ax.set_ylabel('Predator Species')

ani = FuncAnimation(fig, update, frames=range(len(t)), repeat=False, interval=100)
plt.show()
