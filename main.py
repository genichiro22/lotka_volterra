import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from matplotlib.animation import FuncAnimation

# Define parameters
alpha = 0.1
beta = 0.4
gamma = 0.2
delta = 0.2
D1 = 0.05
D2 = 0.1
dt = 0.01
timesteps = 5000

# Define the spatial domain
nx, ny = 500, 500

# Initialize the prey and predator populations
# prey = np.random.rand(nx, ny)
# predator = np.random.rand(nx, ny)
# Define the spatial domain
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
xx, yy = np.meshgrid(x, y)

# Initialize the prey and predator populations with sinusoidal distribution
prey = 1 + 0.5 * np.sin(2*xx*xx) * np.sin(yy)
predator = 1 + 0.5 * np.cos(xx) * np.cos(2*yy*xx)

def compute_laplacian(field):
    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0, 1.0, 0.0]])
    return convolve(field, kernel, mode='wrap')

def update(frame):
    global prey, predator

    for _ in range(5):  # Perform multiple updates per frame for better visualization
        # Compute Laplacians
        laplacian_prey = compute_laplacian(prey)
        laplacian_predator = compute_laplacian(predator)

        # Update prey and predator populations
        prey_next = prey + dt * (D1 * laplacian_prey + alpha * prey - beta * prey * predator)
        predator_next = predator + dt * (D2 * laplacian_predator + delta * prey * predator - gamma * predator)

        # Assign updated populations
        prey, predator = prey_next, predator_next

    # Update the plots
    prey_plot.set_array(prey)
    predator_plot.set_array(predator)

    return prey_plot, predator_plot,

# Set up the visualization and create the animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

prey_plot = ax1.imshow(prey, cmap='viridis', extent=[0, nx, 0, ny], animated=True)
predator_plot = ax2.imshow(predator, cmap='inferno', extent=[0, nx, 0, ny], animated=True)

ax1.set_title('Prey')
ax2.set_title('Predator')

fig.colorbar(prey_plot, ax=ax1, label='Prey population')
fig.colorbar(predator_plot, ax=ax2, label='Predator population')

anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()
