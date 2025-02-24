import numpy as np
import os
import time

from plot import prep_figure, plot_state, plot_finalize

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""


def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law
        pos  is an N x 3 matrix of positions
        mass is an N x 1 vector of masses
        G is Newton's Gravitational constant
        softening is the softening length
        a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def getEnergy(pos, vel, mass, G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum(mass * vel**2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE


def main(
    N=100,
    t=0,
    tEnd=10.0,
    dt=0.01,
    softening=0.1,
    G=1.0,
    plotRealTime=False,
    measureTime=False,
    pos=None,
    vel=None,
):
    """N-body simulation"""

    # Generate Initial Conditions
    np.random.seed(17)  # set the random number generator seed

    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    if pos is None:
        pos = np.random.randn(N, 3)  # randomly selected positions and velocities
    if vel is None:
        vel = np.random.randn(N, 3)

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE, PE = getEnergy(pos, vel, mass, G)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos
    KE_save = np.zeros(Nt + 1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt + 1)
    PE_save[0] = PE
    t_all = np.arange(Nt + 1) * dt

    prep_figure()

    start_time = time.time()
    # Simulation Main Loop
    for i in range(1, Nt + 1):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE, PE = getEnergy(pos, vel, mass, G)

        # save energies, positions for plotting trail
        pos_save[:, :, i] = pos
        KE_save[i] = KE
        PE_save[i] = PE

        # plot in real time
        if plotRealTime:
            plot_state(i, t_all, pos_save, KE_save, PE_save)

    end_time = time.time()
    if measureTime:
        print(f"Execution time: {end_time - start_time} seconds")

    plot_state(i, t_all, pos_save, KE_save, PE_save)
    plot_finalize(
        f"{os.path.dirname(os.path.abspath(__file__))}/nbody_original_{N}_{tEnd}_{dt}_{softening}_{G}.png"
    )

    if pos is not None and vel is not None:
        return pos, vel


if __name__ == "__main__":
    main()
