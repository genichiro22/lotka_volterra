import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
# Define parameters
alpha = 0.1
beta = 0.02
gamma = 0.3
delta = 0.01
D1 = 0.05
D2 = 0.05
dt = 0.01
timesteps = 500

# Define the spatial domain
nx, ny = 100, 100

# Initialize the prey and predator populations
prey = np.random.rand(nx, ny)
predator = np.random.rand(nx, ny)
def compute_laplacian(field):
    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0, 1.0, 0.0]])
    return convolve(field, kernel, mode='wrap')
for t in range(timesteps):
    # Compute Laplacians
    laplacian_prey = compute_laplacian(prey)
    laplacian_predator = compute_laplacian(predator)

    # Update prey and predator populations
    prey_next = prey + dt * (D1 * laplacian_prey + alpha * prey - beta * prey * predator)
    predator_next = predator + dt * (D2 * laplacian_predator + delta * prey * predator - gamma * predator)

    # Assign updated populations
    prey, predator = prey_next, predator_next
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(prey, cmap='viridis', extent=[0, nx, 0, ny])
plt.colorbar(label='Prey population')
plt.title('Prey')

plt.subplot(1, 2, 2)
plt.imshow(predator, cmap='inferno', extent=[0, nx, 0, ny])
plt.colorbar(label='Predator population')
plt.title('Predator')

plt.show()
