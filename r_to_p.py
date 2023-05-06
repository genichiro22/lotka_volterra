import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class SimulationParameters:
    def __init__(self, alphas, betas, psis, deltas, gammas, xis, ks, ls, ds):
        self.alphas = np.array(alphas)
        self.betas = np.array(betas)
        self.psis = np.array(psis)
        self.deltas = np.array(deltas)
        self.gammas = np.array(gammas)
        self.xis = np.array(xis)
        self.ks = np.array(ks)
        self.ls = np.array(ls)
        self.ds = np.array(ds)

class SimulationState:
    def __init__(self, width, height):
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        initial_population = np.exp(-(x**2 + y**2))

        def create_initial_population(freq1, freq2, freq3):
            return 2.1 + np.sin(freq1 * x) * np.cos(freq2 * y) + np.sin(freq3 * x) * np.cos(freq3 * y)

        self.p1 = create_initial_population(freq1=1, freq2=2, freq3=3)
        self.p2 = create_initial_population(freq1=4, freq2=5, freq3=6)
        self.q1 = create_initial_population(freq1=7, freq2=8, freq3=9)
        self.q2 = create_initial_population(freq1=10, freq2=11, freq3=12)
        self.time = 0

    def update(self, params, dt):
        d_p1 = params.alphas[0] * self.p1 * (1 - (self.p1 + params.psis[0, 0] * self.p2) / params.ks[0]) - \
            params.betas[0, 0] * self.p1 * self.q1 - params.betas[0, 1] * self.p1 * self.q2 + \
            params.ds[0, 0] * self.laplacian(self.p1)

        d_p2 = params.alphas[1] * self.p2 * (1 - (self.p2 + params.psis[1, 0] * self.p1) / params.ks[1]) - \
            params.betas[1, 0] * self.p2 * self.q1 - params.betas[1, 1] * self.p2 * self.q2 + \
            params.ds[0, 1] * self.laplacian(self.p2)

        d_q1 = params.deltas[0, 0] * self.p1 * self.q1 + params.deltas[1, 0] * self.p2 * self.q1 - \
            params.gammas[0] * self.q1 * (1 - (self.q1 + params.xis[0, 0] * self.q2) / params.ls[0]) + \
            params.ds[1, 0] * self.laplacian(self.q1)

        d_q2 = params.deltas[0, 1] * self.p1 * self.q2 + params.deltas[1, 1] * self.p2 * self.q2 - \
            params.gammas[1] * self.q2 * (1 - (self.q2 + params.xis[1, 0] * self.q1) / params.ls[1]) + \
            params.ds[1, 1] * self.laplacian(self.q2)

        self.p1 += dt * d_p1
        self.p2 += dt * d_p2
        self.q1 += dt * d_q1
        self.q2 += dt * d_q2
        self.time += dt

    def laplacian(self, grid):
        return np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + \
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid

def main():
    alphas = [0.1, 0.2]
    betas = [[0.3, 0.43], [0.4, 0.41]]
    psis = [[1], [0.3]]
    deltas = [[0.24, 0.21], [0.23, 0.2]]
    gammas = [0.2, 0.2]
    xis = [[0.3], [1]]
    ks = [2000, 1300]
    ls = [65, 100]
    ds = [[0.001, 0.003], [0.002, 0.0015]]

    params = SimulationParameters(alphas, betas, psis, deltas, gammas, xis, ks, ls, ds)

    width, height = 100, 100
    state = SimulationState(width, height)
    # Initialize the state with initial population distributions

    dt = 0.01  # Time step for the simulation
    num_iterations = 30000  # Number of iterations for the simulation
    p1_populations = []
    p2_populations = []
    q1_populations = []
    q2_populations = []
    for _ in range(num_iterations):
        state.update(params, dt)
        # Optionally, output the state or visualize it
        # Record the population data
        p1_populations.append(np.sum(state.p1))
        p2_populations.append(np.sum(state.p2))
        q1_populations.append(np.sum(state.q1))
        q2_populations.append(np.sum(state.q2))
    # Plot the time series data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot the prey populations on the first subplot
    ax1.plot(p1_populations, label='P1')
    ax1.plot(p2_populations, label='P2')
    ax1.set_ylabel('Prey Population')
    ax1.legend()

    # Plot the predator populations on the second subplot
    ax2.plot(q1_populations, label='Q1')
    ax2.plot(q2_populations, label='Q2')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Predator Population')
    ax2.legend()

    plt.show()
if __name__ == "__main__":
    main()
