import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LotkaVolterra2D:
    def __init__(self, params, initial_conditions, diffusion_coeffs, dx, dy, grid_size):
        self.params = params
        self.state = initial_conditions
        self.diffusion_coeffs = diffusion_coeffs
        self.dx = dx
        self.dy = dy
        self.grid_size = grid_size

    def deriv(self, t, y):
        P = y[:3*self.grid_size**2].reshape((3, self.grid_size, self.grid_size))
        Q = y[3*self.grid_size**2:].reshape((self.grid_size, self.grid_size))

        alpha, psi, beta, delta, gamma, K, L = self.params

        dP = alpha[:, None, None]*P*(1 - (P + np.tensordot(psi, P, axes=1))/K[:, None, None]) - beta[:, None, None]*P*Q
        dQ = np.sum(delta[:, None, None]*P, axis=0)*Q - gamma*Q*(1 - Q/L)

        dP += self.diffusion(P, self.diffusion_coeffs[0])
        dQ += self.diffusion(Q, self.diffusion_coeffs[1], True)

        return np.concatenate([dP.ravel(), dQ.ravel()])

    def diffusion(self, arr, D, is_Q=False):
        laplacian = (np.roll(arr, 1, axis=-1) + np.roll(arr, -1, axis=-1) - 2*arr) / self.dx**2 + \
                    (np.roll(arr, 1, axis=-2) + np.roll(arr, -1, axis=-2) - 2*arr) / self.dy**2
        if is_Q:
            return D * laplacian
        else:
            return D[:, None, None] * laplacian

    def run(self, t_span, dt):
        print(t_span)
        sol = solve_ivp(self.deriv, t_span, self.state, t_eval=np.arange(t_span[0], t_span[1], dt))
        self.state = sol.y[:,-1]
        return sol

# Usage:
alpha = np.array([0.1, 0.2, 0.3])
psi = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.3], [0.3, 0.2, 0.1]])
beta = np.array([0.1, 0.2, 0.3])
delta = np.array([0.1, 0.2, 0.3])
gamma = 0.1
K = np.array([100, 200, 300])
L = 1000
params = [alpha, psi, beta, delta, gamma, K, L]

P = np.random.uniform(0, 1, size=(3, 100, 100))
Q = np.random.uniform(0, 1, size=(100, 100))
initial_conditions = np.concatenate([P.ravel(), Q.ravel()])

D_P = np.array([0.01, 0.02, 0.03])
D_Q = 0.01
diffusion_coeffs = [D_P, D_Q]

dx = dy = 0.01

grid_size = 100

# Run the simulation
lv = LotkaVolterra2D(params, initial_conditions, diffusion_coeffs, dx, dy, grid_size)
sol = lv.run([0, 1], 0.01)

# Calculate the total population of each species at each time point
total_population = sol.y.reshape((4, -1, grid_size, grid_size)).sum(axis=(2, 3))

# Plot the total population over time
plt.figure(figsize=(10, 6))
for i, species in enumerate(['P1', 'P2', 'P3', 'Q']):
    plt.plot(sol.t, total_population[i], label=species)
plt.xlabel('Time')
plt.ylabel('Total Population')
plt.legend()
plt.grid(True)
plt.show()
