import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the space randomly with 100 points uniformly distributed
def initialize_space(num_points=100, space_size=1.0):
    points = np.random.uniform(low=0, high=space_size, size=(num_points, 2))
    return points

# Compute gravitational forces on each point from every other point (with softening)
def compute_gravity(points, G=1.0, softening=1e-2):
    num_points = points.shape[0]
    forces = np.zeros_like(points)
    for i in range(num_points):
        diff = points - points[i]
        dist_sq = np.sum(diff**2, axis=1) + softening**2
        inv_dist3 = dist_sq**(-1.5)
        inv_dist3[i] = 0  # no self-force
        forces[i] = G * np.sum(diff * inv_dist3[:, np.newaxis], axis=0)
    return forces

# Update points positions based on forces and a timestep (Euler integration)
def update_positions(points, forces, dt=0.01):
    velocities = forces * dt  # assuming unit mass
    points += velocities * dt
    return points

# Compute density using a simple 2D histogram
def compute_density(points, bins=20, space_size=1.0):
    H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=[[0, space_size], [0, space_size]])
    density = np.sum(H) / (bins * bins)  # average density
    return density

# Animate the space evolution under gravity
def animate_space(num_points=100, num_steps=1000, dt=0.01, space_size=1.0):
    points = initialize_space(num_points, space_size)

    fig, ax = plt.subplots()
    scat = ax.scatter(points[:, 0], points[:, 1], s=10)
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_title('Gravity Simulation with Density')

    density_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        forces = compute_gravity(points)
        update_positions(points, forces, dt)
        scat.set_offsets(points)
        density = compute_density(points, space_size=space_size)
        density_text.set_text(f'Density: {density:.3f}')
        return scat, density_text

    ani = FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
    plt.show()

# Run the animation simulation
animate_space()
