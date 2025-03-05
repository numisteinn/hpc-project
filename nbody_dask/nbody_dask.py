"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

import time
import numpy as np
import dask.array as da
from plot import prep_figure, plot_state


def get_acc_dask(pos, mass, G, softening):
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    dx, dy, dz = x.T - x, y.T - y, z.T - z

    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3 = da.where(inv_r3 > 0, inv_r3 ** (-1.5), 0)

    ax = G * da.matmul(dx * inv_r3, mass)
    ay = G * da.matmul(dy * inv_r3, mass)
    az = G * da.matmul(dz * inv_r3, mass)

    return da.hstack((ax, ay, az))


def get_energy_dask(pos, vel, mass, G):
    KE = 0.5 * da.sum(da.sum(mass * vel**2))

    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    dx, dy, dz = x.T - x, y.T - y, z.T - z

    inv_r = da.sqrt(dx**2 + dy**2 + dz**2)
    epsilon = 1e-5
    inv_r = da.where(inv_r > 0, 1.0 / (inv_r + epsilon), 0)

    PE = G * da.sum(da.sum(da.triu(-(mass * mass.T) * inv_r, 1)))

    return KE.compute(), PE.compute()


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
    np.random.seed(17)
    mass = da.from_array(20.0 * np.ones((N, 1)) / N)
    pos = da.from_array(np.random.randn(N, 3) if pos is None else pos)
    vel = da.from_array(np.random.randn(N, 3) if vel is None else vel)
    vel -= da.mean(mass * vel, axis=0) / da.mean(mass)

    acc = get_acc_dask(pos, mass, G, softening)
    KE, PE = get_energy_dask(pos, vel, mass, G)

    Nt = int(np.ceil(t_end / dt))
    pos_save = np.zeros((N, 3, Nt + 1))
    KE_save, PE_save = np.zeros(Nt + 1), np.zeros(Nt + 1)
    pos_save[:, :, 0], KE_save[0], PE_save[0] = pos.compute(), KE, PE
    t_all = np.arange(Nt + 1) * dt

    prep_figure()
    start_time = time.time()

    for i in range(1, Nt + 1):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = get_acc_dask(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        KE, PE = get_energy_dask(pos, vel, mass, G)
        pos_save[:, :, i], KE_save[i], PE_save[i] = pos.compute(), KE, PE

        if plot_real_time:
            plot_state(i, t_all, pos_save, KE_save, PE_save)

    end_time = time.time()
    if measure_time:
        print(f"Execution time: {end_time - start_time} seconds")

    plot_state(i, t_all, pos_save, KE_save, PE_save)
    return pos.compute(), vel.compute(), KE_save, PE_save


if __name__ == "__main__":
    main()
