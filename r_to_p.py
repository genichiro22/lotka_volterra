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
            return 2.01 + np.sin(freq1 * x) * np.cos(freq2 * y) + np.sin(freq3 * x) * np.cos(freq3 * y)

        self.p1 = create_initial_population(freq1=1, freq2=2, freq3=4)*0
        self.p2 = create_initial_population(freq1=4, freq2=7, freq3=6)*0+1
        self.q1 = create_initial_population(freq1=7, freq2=8, freq3=2)*0
        self.q2 = create_initial_population(freq1=3, freq2=1, freq3=8)*0+1
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
    alphas = [0.2, 0.2]
    betas = [[0.3, 0.3], [0.3, 0.3]]
    psis = [[1], [0.3]]
    deltas = [[0.2, 0.20], [0.2, 0.2]]
    gammas = [0.2, 0.2]
    xis = [[0.3], [1]]
    ks = [2, 1.3]
    ls = [6.5, 10.0]
    ds = [[0.001, 0.003], [0.002, 0.0015]]

    params = SimulationParameters(alphas, betas, psis, deltas, gammas, xis, ks, ls, ds)

    width, height = 1, 1
    state = SimulationState(width, height)
    # Initialize the state with initial population distributions

    dt = 0.01  # Time step for the simulation
    num_iterations = 3000000  # Number of iterations for the simulation

    # Create empty lists to store the population data for each species
    p1_populations = []
    p2_populations = []
    q1_populations = []
    q2_populations = []

    # Display interval for visualizations
    display_interval = 1000

    # Create the initial plots and set up the figure layout
    # plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    # Set up the time series plots
    p1_line, = axs[0, 0].plot([], [], label='P1')
    p2_line, = axs[0, 0].plot([], [], label='P2')
    axs[0, 0].set_xlabel('Time step')
    axs[0, 0].set_ylabel('Prey Population')
    axs[0, 0].legend()

    q1_line, = axs[1, 0].plot([], [], label='Q1')
    q2_line, = axs[1, 0].plot([], [], label='Q2')
    axs[1, 0].set_xlabel('Time step')
    axs[1, 0].set_ylabel('Predator Population')
    axs[1, 0].legend()

    # Set up the distribution colormap plots
    images = []
    for idx, (title, data) in enumerate([('P1', state.p1), ('P2', state.p2), ('Q1', state.q1), ('Q2', state.q2)]):
        r, c = divmod(idx, 2)
        c += 1
        print(idx, r, c)
        im = axs[r, c].imshow(data, cmap='viridis', vmin=0, vmax=data.max())
        axs[r, c].set_title(title)
        fig.colorbar(im, ax=axs[r, c])
        images.append(im)

    plt.tight_layout()

    for i in range(num_iterations):
        state.update(params, dt)

        # Record the population data
        p1_populations.append(np.sum(state.p1))
        p2_populations.append(np.sum(state.p2))
        q1_populations.append(np.sum(state.q1))
        q2_populations.append(np.sum(state.q2))
        # print(p1_populations)

        # Update the plots directly every display_interval iterations
        if i % display_interval == 0:
            # Update the time series data
            p1_line.set_xdata(np.arange(len(p1_populations)))
            p1_line.set_ydata(p1_populations)
            p2_line.set_xdata(np.arange(len(p2_populations)))
            p2_line.set_ydata(p2_populations)
            q1_line.set_xdata(np.arange(len(q1_populations)))
            q1_line.set_ydata(q1_populations)
            q2_line.set_xdata(np.arange(len(q2_populations)))
            q2_line.set_ydata(q2_populations)
            # axs[0, 0].plot()
            # plt.draw()

            # Update the time series x-axis limits
            axs[0, 0].set_xlim(0, len(p1_populations))
            axs[1, 0].set_xlim(0, len(q1_populations))
            axs[0, 0].set_ylim(0, max(p1_populations+p2_populations))
            axs[1, 0].set_ylim(0, max(q1_populations+q2_populations))


            fig.canvas.draw()
            # fig.canvas.flush_events()
            # Update the distribution colormaps
            for idx, data in enumerate([state.p1, state.p2, state.q1, state.q2]):
                images[idx].set_data(data)
                images[idx].set_clim(vmin=0, vmax=data.max())

            plt.pause(0.001)
            # fig.clear()

    plt.show()
if __name__ == "__main__":
    main()
