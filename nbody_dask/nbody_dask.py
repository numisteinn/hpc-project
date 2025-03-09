"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

import numpy as np
import dask.array as da


def compute_acc_chunk(pos_target, pos_source, mass_source, G=1.0, softening=0.1):
    # Compute pairwise differences: result shape (n_target, n_source, 3)
    pos_source = np.vstack(
        pos_source
    )  # Convert list of arrays into single (N_source, 3) array
    mass_source = np.vstack(mass_source)
    pos_target = np.vstack(pos_target)
    diff = pos_source[None, :, :] - pos_target[:, None, :]

    # Compute squared distances with softening: (n_target, n_source)
    dist_sq = np.sum(diff**2, axis=2) + softening**2

    # Compute inverse distance cubed: (n_target, n_source)
    inv_dist3 = dist_sq ** (-1.5)

    # Compute contributions: multiply differences by inv_dist3 and mass_source
    # mass_source: shape (n_source, 1) broadcasts correctly to (n_target, n_source)
    acc_contrib = G * (diff * inv_dist3[:, :, None]) * mass_source[None, :, :]
    return acc_contrib


def get_acc_dask(pos, mass, G, softening):
    distances = da.blockwise(
        compute_acc_chunk,
        "ijk",  # output indices: i (first axis), j (second axis)
        pos,
        "ik",  # first input: positions with indices i and k
        pos,
        "jk",  # second input: positions with indices j and k
        mass,
        "j1",  # third input: mass with index j
        softening=softening,  # pass softening as a keyword argument
        G=G,
        dtype=float,
        adjust_chunks={"i": pos.chunks[0], "j": pos.chunks[0]},
    )
    return da.sum(distances, axis=1)


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
    save_plot=True,
    plot_real_time=False,
    pos=None,
    vel=None,
):
    np.random.seed(17)
    mass = da.from_array(20.0 * np.ones((N, 1)) / N, chunks=(N // 4, 1))
    pos = (
        da.from_array(np.random.randn(N, 3), chunks=(N // 4, 3)) if pos is None else pos
    )
    vel = (
        da.from_array(np.random.randn(N, 3), chunks=(N // 4, 3)) if vel is None else vel
    )
    vel -= da.mean(mass * vel, axis=0) / da.mean(mass)
    acc = get_acc_dask(pos, mass, G, softening)
    Nt = int(np.ceil(t_end / dt))
    for i in range(1, Nt + 1):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = get_acc_dask(pos, mass, G, softening)
        vel += acc * dt / 2

    KE, PE = get_energy_dask(pos, vel, mass, G)
    out = pos.compute(), vel.compute(), KE.compute(), PE.compute()
    return out


if __name__ == "__main__":
    main()
