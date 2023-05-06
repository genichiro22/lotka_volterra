import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from matplotlib.colors import LogNorm

# Parameters
alpha1, alpha2, beta11, beta12, beta21, beta22 = 0.1, 0.1, 0.3, 0.43, 0.4, 0.41
delta11, delta12, delta21, delta22, gamma1, gamma2 = 0.24, 0.21, 0.23, 0.2, 0.2, 0.2
psi12, psi21, K1, K2 = 1, 0.3, 2000, 1300
xi12, xi21, L1, L2 = 0.3, 1, 65, 100 # 1, 0.3, 100, 65
D_P1, D_P2, D_Q1, D_Q2 = 0.001, 0.003, 0.002, 0.0015
dt = 0.05
steps = 10000

# Grid size
nx, ny = 500, 500
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
xx, yy = np.meshgrid(x, y)

# Initialize arrays
P1 = 1 + 0.5 * np.sin(2*xx*xx) * np.sin(yy)
P2 = 1 + 0.5 * np.sin(2*(6-xx)*(6-xx)) * np.sin((6-yy))
Q1 = 1 + 0.5 * np.cos(xx) * np.cos(2*yy*xx)
Q2 = 1 + 0.5 * np.cos(6-xx) * np.cos(2*(6-yy)*(xx))

# Laplacian function
def laplacian(Z):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve(Z, kernel, mode='wrap')

# Update function for animation
def update(num):
    global P1, P2, Q1, Q2
    for _ in range(10):
        P1_lap = laplacian(P1)
        P2_lap = laplacian(P2)
        Q1_lap = laplacian(Q1)
        Q2_lap = laplacian(Q2)

        P1_new = P1 + dt * (alpha1 * P1 * (1 - (P1 + psi12 * P2) / K1) - beta11 * P1 * Q1 - beta12 * P1 * Q2 + D_P1 * P1_lap)
        P2_new = P2 + dt * (alpha2 * P2 * (1 - (P2 + psi21 * P1) / K2) - beta21 * P2 * Q1 - beta22 * P2 * Q2 + D_P2 * P2_lap)
        Q1_new = Q1 + dt * (delta11 * P1 * Q1 + delta21 * P2 * Q1 - gamma1 * Q1 * (1 - (Q1 + xi12 * Q2) / L1) + D_Q1 * Q1_lap)
        Q2_new = Q2 + dt * (delta12 * P1 * Q2 + delta22 * P2 * Q2 - gamma2 * Q2 * (1 - (Q2 + xi21 * Q1) / L2) + D_Q2 * Q2_lap)

        P1, P2, Q1, Q2 = P1_new, P2_new, Q1_new, Q2_new
        # print(P1.sum(), P2.sum(), Q1.sum(), Q2.sum())

    im_P1.set_array(P1)
    im_P2.set_array(P2)
    im_Q1.set_array(Q1)
    im_Q2.set_array(Q2)

    sum_P1_text.set_text(f'P1 sum: {P1.sum():.2f}')
    sum_P2_text.set_text(f'P2 sum: {P2.sum():.2f}')
    sum_Q1_text.set_text(f'Q1 sum: {Q1.sum():.2f}')
    sum_Q2_text.set_text(f'Q2 sum: {Q2.sum():.2f}')
    sum_text.set_text(f'P1 sum: {P1.sum():.2f}\nP2 sum: {P2.sum():.2f}\nQ1 sum: {Q1.sum():.2f}\nQ2 sum: {Q2.sum():.2f}')


    return [im_P1, im_P2, im_Q1, im_Q2, sum_P1_text, sum_P2_text, sum_Q1_text, sum_Q2_text, sum_text]

vmin_P1, vmax_P1 = 1e-5, 10
vmin_P2, vmax_P2 = 1e-5, 10
vmin_Q1, vmax_Q1 = 1e-5, 10
vmin_Q2, vmax_Q2 = 1e-5, 10

# Set up the plot
fig, ((ax_P1, ax_P2), (ax_Q1, ax_Q2)) = plt.subplots(2, 2, figsize=(15, 10))

interpolation = None

im_P1 = ax_P1.imshow(P1, cmap='viridis', interpolation=interpolation, extent=[0, nx, 0, ny],
                     norm=LogNorm(vmin=vmin_P1, vmax=vmax_P1))
ax_P1.set_title('P1 Concentration')
plt.colorbar(im_P1, ax=ax_P1)
sum_P1_text = ax_P1.text(5, 5, '', color='white', fontsize=12)

im_P2 = ax_P2.imshow(P2, cmap='viridis', interpolation=interpolation, extent=[0, nx, 0, ny],
                     norm=LogNorm(vmin=vmin_P2, vmax=vmax_P2))
ax_P2.set_title('P2 Concentration')
plt.colorbar(im_P2, ax=ax_P2)
sum_P2_text = ax_P2.text(5, 5, '', color='white', fontsize=12)

im_Q1 = ax_Q1.imshow(Q1, cmap='plasma', interpolation=interpolation, extent=[0, nx, 0, ny],
                     norm=LogNorm(vmin=vmin_Q1, vmax=vmax_Q1))
ax_Q1.set_title('Q1 Concentration')
plt.colorbar(im_Q1, ax=ax_Q1)
sum_Q1_text = ax_Q1.text(5, 5, '', color='white', fontsize=12)

im_Q2 = ax_Q2.imshow(Q2, cmap='plasma', interpolation=interpolation, extent=[0, nx, 0, ny],
                     norm=LogNorm(vmin=vmin_Q2, vmax=vmax_Q2))
ax_Q2.set_title('Q2 Concentration')
plt.colorbar(im_Q2, ax=ax_Q2)
sum_Q2_text = ax_Q2.text(5, 5, '', color='white', fontsize=12)

ax_text = fig.add_axes([0.65, 0.25, 0.1, 0.1])
ax_text.axis('off')
sum_text = ax_text.text(0, 0.8, '', fontsize=12)

# Animation function
ani = FuncAnimation(fig, update, frames=100, interval=100, repeat=True, blit=True)

# Show the animation
plt.show()
