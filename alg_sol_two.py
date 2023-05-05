import sympy as sp
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
def objective(params):
    return -np.sum(steady(params))
def constraint(params):
    return np.min(steady(params))

# Set initial populations and parameters
initial_populations = [40, 50, 10, 20]
params = [1.2, 1.1, 0.3, 0.4, 0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.3, 0.3, 100, 100, 0.2, 0.2, 50, 50]
t = np.linspace(0, 1, 1)

constraints = {'type': 'ineq', 'fun': constraint}
bounds = [(0.01, 2)]*14 + [(1, 200)]*2 + [(0.01, 2)]*2 + [(1, 200)]*2
initial_guess = params
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

if result.success:
    optimal_params = result.x
    print("Optimal parameters found:", optimal_params)
    print("Steady values:", steady(optimal_params))
else:
    print("No solution found.")

alpha1, alpha2, beta11, beta12, beta21, beta22, delta11, delta12, delta21, delta22, gamma1, gamma2, psi12, psi21, K1, K2, xi12, xi21, L1, L2, P1, P2, Q1, Q2 = sp.symbols(
    'alpha1 alpha2 beta11 beta12 beta21 beta22 delta11 delta12 delta21 delta22 gamma1 gamma2 psi12 psi21 K1 K2 xi12 xi21 L1 L2 P1 P2 Q1 Q2'
)

coeff_matrix_np = np.array(
    [
        [-alpha1/K1, -alpha1/K1*psi12, -beta11, -beta12],
        [-alpha2/K2*psi21, -alpha2/K2, -beta21, -beta22],
        [delta11, delta21, gamma1/L1, gamma1/L1*xi12],
        [delta12, delta22, gamma2/L2*xi21, gamma2/L2],
    ],
    dtype=object
)
const_matrix_np = np.array([-alpha1, -alpha2, gamma1, gamma2])

coeff_matrix_sp = sp.Matrix(coeff_matrix_np.shape[0], coeff_matrix_np.shape[1], lambda i, j: coeff_matrix_np[i, j])
const_matrix = sp.Matrix(const_matrix_np)
solution = sp.linsolve((coeff_matrix_sp, const_matrix), P1, P2, Q1, Q2)
print(f"Solution: {solution}")

