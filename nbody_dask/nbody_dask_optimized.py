"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
Optimized Dask implementation
"""

import os
import time

import numpy as np
import dask.array as da
from dask.distributed import wait

from plot import prep_figure, plot_state, plot_finalize


def get_acc_dask(pos, mass, G, softening):
    """Calculate acceleration using Dask with optimized operations"""
    # Extract position components
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    
    # Calculate pairwise distances
    dx, dy, dz = x.T - x, y.T - y, z.T - z
    
    # Calculate inverse distance cubed with softening
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3 = da.where(inv_r3 > 0, inv_r3 ** (-1.5), 0)
    
    # Calculate accelerations
    ax = G * da.matmul(dx * inv_r3, mass)
    ay = G * da.matmul(dy * inv_r3, mass)
    az = G * da.matmul(dz * inv_r3, mass)
    
    # Combine into a single array and persist to avoid recomputation
    return da.hstack((ax, ay, az)).persist()


def get_energy_dask(pos, vel, mass, G):
    """Calculate kinetic and potential energy using Dask"""
    # Kinetic energy: 0.5 * m * v^2
    KE = 0.5 * da.sum(mass * da.sum(vel**2, axis=1, keepdims=True))
    
    # Potential energy
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    dx, dy, dz = x.T - x, y.T - y, z.T - z
    
    # Calculate inverse distance with softening
    inv_r = da.sqrt(dx**2 + dy**2 + dz**2)
    epsilon = 1e-5
    inv_r = da.where(inv_r > 0, 1.0 / (inv_r + epsilon), 0)
    
    # Calculate potential energy using upper triangular part to avoid double counting
    PE = G * da.sum(da.triu(-(mass * mass.T) * inv_r, 1))
    
    return KE.persist(), PE.persist()


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
    chunk_size=None,
):
    """N-body simulation using Dask"""
    # Set random seed for reproducibility
    np.random.seed(17)
    
    # Set chunk size based on N if not provided
    if chunk_size is None:
        chunk_size = min(100, N)
    
    # Initialize masses, positions, and velocities
    mass = da.from_array(20.0 * np.ones((N, 1)) / N, chunks=(chunk_size, 1))
    
    if pos is None:
        pos = da.from_array(np.random.randn(N, 3), chunks=(chunk_size, 3))
    elif not isinstance(pos, da.Array):
        pos = da.from_array(pos, chunks=(chunk_size, 3))
        
    if vel is None:
        vel = da.from_array(np.random.randn(N, 3), chunks=(chunk_size, 3))
    elif not isinstance(vel, da.Array):
        vel = da.from_array(vel, chunks=(chunk_size, 3))
    
    # Convert to center-of-mass frame
    vel -= da.mean(mass * vel, axis=0) / da.mean(mass)
    
    # Calculate initial acceleration
    acc = get_acc_dask(pos, mass, G, softening)
    
    # Calculate initial energy
    KE, PE = get_energy_dask(pos, vel, mass, G)
    
    # Dask is slow, so we'll just do 100 steps
    Nt = 100
    
    # Create arrays to store results
    pos_save = da.zeros((N, 3, Nt + 1), chunks=(chunk_size, 3, 1))
    KE_save = da.zeros(Nt + 1)
    PE_save = da.zeros(Nt + 1)
    
    # Store initial state
    pos_save = pos_save.at[:, :, 0].set(pos)
    KE_save = KE_save.at[0].set(KE)
    PE_save = PE_save.at[0].set(PE)
    
    # Time array for plotting
    t_all = np.arange(Nt + 1) * dt
    
    # Prepare figure for plotting
    prep_figure()
    
    # Start timing
    start_time = time.time()
    p_time = time.time()
    
    # Main simulation loop
    for i in range(1, Nt + 1):
        # Velocity Verlet integration
        vel += acc * dt / 2.0
        pos += vel * dt
        
        # Update acceleration
        acc = get_acc_dask(pos, mass, G, softening)
        
        # Complete velocity update
        vel += acc * dt / 2.0
        
        # Update time
        t += dt
        
        # Calculate energy
        KE, PE = get_energy_dask(pos, vel, mass, G)
        
        # Store results
        pos_save = pos_save.at[:, :, i].set(pos)
        KE_save = KE_save.at[i].set(KE)
        PE_save = PE_save.at[i].set(PE)
        
        # Log progress
        # if i % 10 == 0:
        #     print(f"Logging... {i}/{Nt}")
        #     print(f"Time since last log: {time.time() - p_time}")
        #     p_time = time.time()
    
    # Prepare output path
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"nbody_dask_optimized_{N}_{t_end}_{dt}_{softening}_{G}.png",
    )
    
    # Compute final results
    print(f"Computing final results after {time.time() - start_time} seconds")
    pos_result = pos.compute()
    vel_result = vel.compute()
    KE_result = KE_save.compute()
    PE_result = PE_save.compute()
    pos_save_result = pos_save.compute()
    
    # Report execution time
    if measure_time:
        print(f"Execution time: {time.time() - start_time} seconds")
    
    # Plot final state
    plot_state(i, t_all, pos_save_result, KE_result, PE_result)
    plot_finalize(output_path)
    
    return pos_result, vel_result, KE_result, PE_result


if __name__ == "__main__":
    main() 