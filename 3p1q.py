import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

class Parameters:
    def __init__(self, alpha, beta, gamma, delta, psi, k, l, d_p, d_q):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.psi = psi
        self.k = k
        self.l = l
        self.d_p = d_p
        self.d_q = d_q

class State:
    def __init__(self, shape):
        self.p = np.zeros((3, *shape))
        self.q = np.zeros(shape)

    def laplacian(self, grid):
        return np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid

    def calculate_step(self, params, dt):
        laplacian_p = np.array([self.laplacian(self.p[i]) for i in range(3)])
        laplacian_q = self.laplacian(self.q)

        # Implement the modified Lotka-Volterra system equations and update the current state.
        for i in range(3):
            # growth_term = params.alpha[i] * self.p[i] * (1 - (self.p[i] + np.sum(params.psi[i] * self.p, axis=0)) / params.k[i])
            growth_term = params.alpha[i] * self.p[i] * (1 - (self.p[i] + np.sum(params.psi[i][:, np.newaxis, np.newaxis] * self.p, axis=0)) / params.k[i])
            # growth_term = params.alpha[i] * self.p[i]

            predation_term = params.beta[i] * self.p[i] * self.q
            diffusion_term = params.d_p[i] * laplacian_p[i]
            self.p[i] += dt * (growth_term - predation_term + diffusion_term)

        predation_term = np.sum(params.delta[:, np.newaxis, np.newaxis] * self.p * self.q, axis=0)
        carrying_capacity_term = params.gamma * self.q * (1 - self.q / params.l)
        diffusion_term = params.d_q * laplacian_q
        self.q += dt * (predation_term - carrying_capacity_term + diffusion_term)

def find_positive_k(alpha, k, tol=1e-6, max_iter=10000):
    # Iterate until all ingredients are positive or maximum number of iterations is reached
    for i in range(max_iter):
        # Create A by adding the identity matrix to alpha
        A = alpha + np.identity(len(k))

        # Calculate X using k
        X = np.matmul(np.linalg.inv(A), k)

        # Check if all ingredients are positive
        if all(X > 0):
            return k

        # Adjust k based on the sign of the ingredients
        for j in range(len(k)):
            if X[j] <= 0:
                k[j] *= -1
                # If changing the sign doesn't help, try a small positive value
                if np.matmul(np.linalg.inv(A), k)[j] <= 0:
                    k[j] = 0.1

        # Check if the difference in ingredients between iterations is smaller than the tolerance level
        if i > 0 and np.allclose(X, prev_X, rtol=0, atol=tol):
            return k

        # Save the current X for the next iteration
        prev_X = X.copy()

    # If maximum number of iterations is reached and no solution is found, return None
    return None

from scipy.optimize import minimize

def find_positive_k2(alpha, k0, tol=1e-7, max_iter=10000):
    # Define a function that computes the negative sum of the logarithms of the ingredients
    def neg_log_ingredient(k):
        A = alpha + np.identity(len(k))
        X = np.matmul(np.linalg.inv(A), k)
        return -np.sum(np.log(X))

    # Define the bounds for k (all values must be positive)
    bounds = [(0, None)] * len(k0)

    # Define the constraint that all ingredients must be positive
    constraint = {'type': 'ineq', 'fun': lambda k: k-1e2}

    # Iterate until a positive solution is found or maximum number of iterations is reached
    for i in range(max_iter):
        # Use the minimize function to find the optimal k
        res = minimize(neg_log_ingredient, k0, bounds=bounds, constraints=constraint)

        # Check if all ingredients are positive
        A = alpha + np.identity(len(k0))
        X = np.matmul(np.linalg.inv(A), res.x)
        if all(X > 0):
            return res.x

        # Update k0 to the solution of the previous iteration
        k0 = res.x

        # Check if the difference in ingredients between iterations is smaller than the tolerance level
        if i > 0 and np.allclose(X, prev_X, rtol=0, atol=tol):
            return res.x

        # Save the current X for the next iteration
        prev_X = X.copy()

    # If maximum number of iterations is reached and no solution is found, return None
    return None
def find_coefficients(X1, X2, X3):
    c = [0, 0, 0]  # No specific objective function; we just want a feasible solution
    A = np.column_stack((X1, X2, X3))
    b = [1, 1, 1]
    bounds = [(None, None), (None, None), (None, None)]  # No non-negativity constraint on c1, c2, and c3
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if res.success:
        c1, c2, c3 = res.x
        k = c1 * X1 + c2 * X2 + c3 * X3
        return c1, c2, c3, k
    else:
        return None, None, None, None
def main():
    # Initialize the model parameters.
    alpha = np.array([0.2, 0.2, 0.2])
    beta = np.array([0.1, 0.1, 0.1])
    gamma = 0.5
    delta = np.array([0.1, 0.2, 0.3])
    psi = np.array([
        [-1, 0.3, 0.9],
        [0.3, -1, 0.1],
        [0.1, 0.9, -1]
    ])
    k = np.array([100, 200, 130])
    l = 500
    d_p = np.array([0.04, 0.05, 0.06])
    d_q = 0.01

    A = psi + np.identity(3)
    lmd, X = np.linalg.eig(A)
    print(f"Eigenvalues: {lmd}")
    print(f"Eigenvectors: {X}")
    print(f"Stable population: {np.linalg.solve(A, k)}")
    k = find_positive_k2(psi, k0=k)
    print(k)
    # print(find_coefficients(X[:,0], X[:,1], X[:,2]))
    # c1,c2,c3,k=find_coefficients(X[:,0], X[:,1], X[:,2])
    print(f"Stable population: {np.linalg.solve(A, k)}")
    
    params = Parameters(alpha, beta, gamma, delta, psi, k, l, d_p, d_q)
    width, heights = 100,100
    state = State((width, heights))

    # Set initial conditions.
    # state.p[:, 10:20, 10:20] = np.array([[[50] * 10] * 10, [[100] * 10] * 10, [[150] * 10] * 10])
    state.p = np.random.rand(3,width, heights)

    # state.q[10:20, 10:20] = 250
    state.q = np.ones((width, heights))

    dt = 0.01
    num_iterations = 20000



    # # Time loop
    # for t in range(num_iterations):
    #     state.calculate_step(params, dt)

    #     # Implement
    #     # Implement boundary conditions and/or data output as needed.
    #     if t % 1000 == 0:
    #         fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    #         fig.suptitle(f"Time step: {t}")

    #         for i in range(3):
    #             axes[i // 2, i % 2].imshow(state.p[i], cmap="viridis", vmin=0, vmax=np.max(state.p))
    #             axes[i // 2, i % 2].set_title(f"P{i + 1}")

    #         axes[1, 1].imshow(state.q, cmap="viridis", vmin=0, vmax=np.max(state.q))
    #         axes[1, 1].set_title("Q")

    #         plt.savefig(f"output_{t:04d}.png")
    #         plt.close(fig)

    avg_p1 = np.zeros(num_iterations)
    avg_p2 = np.zeros(num_iterations)
    avg_p3 = np.zeros(num_iterations)
    avg_q = np.zeros(num_iterations)

    # Time loop
    for t in range(num_iterations):
        state.calculate_step(params, dt)

        # Calculate average populations
        avg_p1[t] = np.mean(state.p[0])
        avg_p2[t] = np.mean(state.p[1])
        avg_p3[t] = np.mean(state.p[2])
        avg_q[t] = np.mean(state.q)

        if t%1000==0:
            print(t)
            # print(params.psi)
            # print(params.psi[0][:, np.newaxis, np.newaxis])
            # print(params.psi[1][:, np.newaxis, np.newaxis])
            # print(params.psi[2][:, np.newaxis, np.newaxis])

    # Plot time series line for average populations
    plt.plot(avg_p1, label="P1")
    plt.plot(avg_p2, label="P2")
    plt.plot(avg_p3, label="P3")
    plt.plot(avg_q, label="Q")
    plt.xlabel("Time step")
    plt.ylabel("Average population")
    plt.legend()
    plt.savefig("time_series_line.png")
    plt.show()

if __name__ == "__main__":
    main()
