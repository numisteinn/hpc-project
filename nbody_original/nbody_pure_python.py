import time
import random
import math

import numpy as np
from matplotlib import pyplot as plt


def prep_figure():
    global grid, ax1, ax2
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])


def plot_state(step, t_all, pos_save, KE_save, PE_save):
    plt.sca(ax1)
    plt.cla()
    xx = [[r[0] for r in pp] for pp in pos_save[max(step - 50, 0) : step]]
    yy = [[r[1] for r in pp] for pp in pos_save[max(step - 50, 0) : step]]
    plt.scatter(xx, yy, s=1, color=[0.7, 0.7, 1])
    a = [r[0] for r in pos_save[step]]
    b = [r[1] for r in pos_save[step]]
    plt.scatter(a, b, s=10, color="blue")
    ax1.set(xlim=(-2, 2), ylim=(-2, 2))
    ax1.set_aspect("equal", "box")
    ax1.set_xticks([-2, -1, 0, 1, 2])
    ax1.set_yticks([-2, -1, 0, 1, 2])

    plt.sca(ax2)
    plt.cla()
    plt.scatter(t_all, KE_save, color="red", s=1, label="KE")
    plt.scatter(t_all, PE_save, color="blue", s=1, label="PE")
    total_E = [KE_save[i] + PE_save[i] for i in range(len(KE_save))]
    plt.scatter(t_all, total_E, color="black", s=1, label="Etot")
    ax2.set(xlim=(0, t_all[-1]), ylim=(-300, 300))
    ax2.set_aspect(0.007)

    plt.pause(0.001)


def get_acc(pos, mass, G, softening):
    """Computes acceleration using direct pairwise interactions."""
    N = len(pos)
    acc = [[0.0, 0.0, 0.0] for _ in range(N)]  # Initialize acceleration array

    for i in range(N):
        for j in range(N):
            if i != j:  # Avoid self-interaction
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dz = pos[j][2] - pos[i][2]

                r2 = dx**2 + dy**2 + dz**2 + softening**2
                inv_r3 = (r2**-1.5) if r2 > 0 else 0.0

                acc[i][0] += G * dx * inv_r3 * mass[j]
                acc[i][1] += G * dy * inv_r3 * mass[j]
                acc[i][2] += G * dz * inv_r3 * mass[j]

    return acc


def get_energy(pos, vel, mass, G):
    """Computes kinetic and potential energy."""
    N = len(pos)
    KE, PE = 0.0, 0.0

    # Kinetic Energy
    for i in range(N):
        v2 = vel[i][0] ** 2 + vel[i][1] ** 2 + vel[i][2] ** 2
        KE += 0.5 * mass[i] * v2

    # Potential Energy
    for i in range(N):
        for j in range(i + 1, N):  # Avoid double-counting
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dz = pos[j][2] - pos[i][2]
            r = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-5  # Avoid singularities
            PE -= G * mass[i] * mass[j] / r

    return KE, PE


def main(
    N=100,
    t=0,
    t_end=10.0,
    dt=0.01,
    softening=0.1,
    G=1.0,
    plot_real_time=False,
    measure_time=False,
    pos=None,
    vel=None,
):
    """Runs the N-body simulation."""
    random.seed(17)

    # Initialize masses, positions, and velocities
    mass = [20.0 / N for _ in range(N)]
    pos = [[random.gauss(0, 1) for _ in range(3)] for _ in range(N)]
    vel = [[random.gauss(0, 1) for _ in range(3)] for _ in range(N)]

    # Remove momentum drift
    mean_vel = [
        sum(vel[i][j] * mass[i] for i in range(N)) / sum(mass) for j in range(3)
    ]
    for i in range(N):
        vel[i] = [vel[i][j] - mean_vel[j] for j in range(3)]

    # Compute initial acceleration
    acc = get_acc(pos, mass, G, softening)

    # Store results

    Nt = int(math.ceil(t_end / dt))
    pos_save = [[[0.0] * 3 for _ in range(N)] for _ in range(Nt + 1)]
    KE_save, PE_save = [0.0] * (Nt + 1), [0.0] * (Nt + 1)
    t_all = np.arange(Nt + 1) * dt

    prep_figure()

    KE, PE = get_energy(pos, vel, mass, G)
    pos_save[0], KE_save[0], PE_save[0] = pos[:], KE, PE

    start_time = time.time()

    for i in range(1, Nt + 1):
        # Velocity Verlet integration
        for j in range(N):
            vel[j] = [vel[j][k] + acc[j][k] * dt / 2.0 for k in range(3)]
            pos[j] = [pos[j][k] + vel[j][k] * dt for k in range(3)]

        acc = get_acc(pos, mass, G, softening)

        for j in range(N):
            vel[j] = [vel[j][k] + acc[j][k] * dt / 2.0 for k in range(3)]

        KE, PE = get_energy(pos, vel, mass, G)
        pos_save[i], KE_save[i], PE_save[i] = pos[:], KE, PE
        # plot in real time
        if plot_real_time:
            plot_state(i, t_all, pos_save, KE_save, PE_save)

    end_time = time.time()
    if measure_time:
        print(f"Execution time: {end_time - start_time:.3f} seconds")

    plot_state(i, t_all, pos_save, KE_save, PE_save)
    # plot_finalize(
    #     f"{os.path.dirname(os.path.abspath(__file__))}/nbody_original_{N}_{t_end}_{dt}_{softening}_{G}.png"
    # )
    return pos, vel, KE_save, PE_save


# Run the simulation
if __name__ == "__main__":
    main(N=100, measure_time=False)
