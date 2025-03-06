"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

import time

import numpy as np
import dask.array as da


def compute_acc_chunk(pos_block, mass, G, softening):
    print(f"Computing acceleration for chunk of shape {pos_block.shape}")
    print(f"Mass shape: {mass.shape}")
    x = pos_block[:, 0:1]
    y = pos_block[:, 1:2]
    z = pos_block[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    m = mass[: pos_block.shape[0], :]
    ax = G * (dx * inv_r3) @ m
    ay = G * (dy * inv_r3) @ m
    az = G * (dz * inv_r3) @ m

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))
    return a


def get_acc_dask(pos, mass, G, softening):
    # TODO: This does not correctly calculate acc, Have to understand better how chunking works.
    return pos.map_overlap(
        lambda p: compute_acc_chunk(p, mass, G, softening),
        depth=(1, 0),
        boundary="reflect",
        dtype=np.float64,
    )


def get_energy_dask(pos, vel, mass, G):
    KE = 0.5 * da.sum(da.sum(mass * vel**2))

    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    dx, dy, dz = x.T - x, y.T - y, z.T - z

    inv_r = da.sqrt(dx**2 + dy**2 + dz**2)
    epsilon = 1e-5
    inv_r = da.where(inv_r > 0, 1.0 / (inv_r + epsilon), 0)

    PE = G * da.sum(da.sum(da.triu(-(mass * mass.T) * inv_r, 1)))

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
    np.random.seed(17)
    # mass = da.from_array(20.0 * np.ones((N, 1)) / N)
    mass = da.from_array(20.0 * np.ones((N, 1)) / N, chunks=(N // 4, 1))
    pos = (
        da.from_array(np.random.randn(N, 3), chunks=(N // 4, 3)) if pos is None else pos
    )
    vel = (
        da.from_array(np.random.randn(N, 3), chunks=(N // 4, 3)) if vel is None else vel
    )
    vel -= da.mean(mass * vel, axis=0) / da.mean(mass)
    acc = get_acc_dask(pos, mass, G, softening)
    print(f"pos shape: {pos.chunks}")
    print(f"Mass shape: {acc.chunks}")
    print(f"Dask Initial accelerations: {acc[:5].compute()}")

    Nt = int(np.ceil(t_end / dt))
    start_time = time.time()
    p_time = time.time()
    # for i in range(1, Nt + 1):
    #     vel += acc * dt / 2.0
    #     pos += vel * dt
    #     acc = get_acc_dask(pos, mass, G, softening)
    #     vel += acc * dt / 2
    #     if i % 10 == 0:
    #         print(f"Logging... {i}/{Nt}")
    #         print(f"Time since last log: {time.time() - p_time}")
    #         p_time = time.time()

    KE, PE = get_energy_dask(pos, vel, mass, G)
    print("Computing...")
    out = pos.compute(), vel.compute(), KE.compute(), PE.compute()
    if measure_time:
        print(
            f"Execution time: {time.time() - start_time} seconds for {Nt} steps and {N} particles"
        )

    return out


if __name__ == "__main__":
    main()
