import numpy as np
import os
import cythonfn
import time

from plot import prep_figure, plot_state, plot_finalize

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""


def main(N=100, t=0, tEnd=10.0, dt=0.01, softening=0.1, G=1.0, plotRealTime=False, measureTime=False, pos=None, vel=None):
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
    acc = cythonfn.getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE, PE = cythonfn.getEnergy(pos, vel, mass, G)

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
        acc = cythonfn.getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE, PE = cythonfn.getEnergy(pos, vel, mass, G)

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


if __name__ == "__main__":
    main()
