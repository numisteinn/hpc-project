import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize

from nbody_dask.nbody_dask import main as dask_main, get_acc_dask, get_energy_dask

def analyze_dask_scaling(n_workers_list, N=500, t_end=1.0):
    """Analyze how Dask scales with different numbers of workers"""
    results = []
    
    for n_workers in n_workers_list:
        print(f"Testing with {n_workers} workers...")
        
        # Start a local Dask cluster with the specified number of workers
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dashboard link: {client.dashboard_link}")
        
        # Generate consistent initial conditions
        np.random.seed(42)
        pos_np = np.random.randn(N, 3).astype(np.float32)
        vel_np = np.random.randn(N, 3).astype(np.float32)
        
        # Create Dask arrays
        pos = da.from_array(pos_np.copy())
        vel = da.from_array(vel_np.copy())
        
        # Run with profiling
        with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:
            with ProgressBar():
                start_time = time.time()
                dask_main(
                    N=N,
                    t=0,
                    t_end=t_end,
                    dt=0.01,
                    softening=0.1,
                    G=1.0,
                    plot_real_time=False,
                    measure_time=False,
                    pos=pos,
                    vel=vel,
                )
                end_time = time.time()
        
        # Save profiling results
        os.makedirs("dask_profiles", exist_ok=True)
        visualize([prof, rprof, cprof], 
                  file_path=f"dask_profiles/profile_N{N}_workers{n_workers}.html",
                  show=False)
        
        # Record results
        results.append({
            "n_workers": n_workers,
            "execution_time": end_time - start_time,
            "N": N
        })
        
        # Close client and cluster
        client.close()
        cluster.close()
    
    return pd.DataFrame(results)

def analyze_task_granularity(chunk_sizes, N=500, t_end=1.0):
    """Analyze how different chunk sizes affect performance"""
    results = []
    
    # Start a local Dask cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    
    for chunk_size in chunk_sizes:
        print(f"Testing with chunk size {chunk_size}...")
        
        # Generate consistent initial conditions
        np.random.seed(42)
        pos_np = np.random.randn(N, 3).astype(np.float32)
        vel_np = np.random.randn(N, 3).astype(np.float32)
        
        # Create Dask arrays with specified chunk size
        pos = da.from_array(pos_np.copy(), chunks=(chunk_size, 3))
        vel = da.from_array(vel_np.copy(), chunks=(chunk_size, 3))
        
        # Run with timing
        start_time = time.time()
        dask_main(
            N=N,
            t=0,
            t_end=t_end,
            dt=0.01,
            softening=0.1,
            G=1.0,
            plot_real_time=False,
            measure_time=False,
            pos=pos,
            vel=vel,
        )
        end_time = time.time()
        
        # Record results
        results.append({
            "chunk_size": chunk_size,
            "execution_time": end_time - start_time,
            "N": N
        })
    
    # Close client and cluster
    client.close()
    cluster.close()
    
    return pd.DataFrame(results)

def analyze_function_performance(N_values, t_end=0.1):
    """Analyze performance of individual functions in the Dask implementation"""
    results = []
    
    for N in N_values:
        print(f"Testing with N={N}...")
        
        # Generate consistent initial conditions
        np.random.seed(42)
        pos_np = np.random.randn(N, 3).astype(np.float32)
        vel_np = np.random.randn(N, 3).astype(np.float32)
        mass_np = 20.0 * np.ones((N, 1)) / N
        
        # Create Dask arrays
        pos = da.from_array(pos_np.copy())
        vel = da.from_array(vel_np.copy())
        mass = da.from_array(mass_np.copy())
        
        # Time get_acc_dask function
        start_time = time.time()
        acc = get_acc_dask(pos, mass, 1.0, 0.1)
        acc.compute()  # Force computation
        acc_time = time.time() - start_time
        
        # Time get_energy_dask function
        start_time = time.time()
        KE, PE = get_energy_dask(pos, vel, mass, 1.0)
        KE.compute(), PE.compute()  # Force computation
        energy_time = time.time() - start_time
        
        # Record results
        results.append({
            "N": N,
            "function": "get_acc_dask",
            "execution_time": acc_time
        })
        results.append({
            "N": N,
            "function": "get_energy_dask",
            "execution_time": energy_time
        })
    
    return pd.DataFrame(results)

def plot_scaling_results(df, output_dir="benchmark_results"):
    """Plot results from scaling analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Worker scaling plot
    if "n_workers" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="n_workers", y="execution_time", marker="o")
        plt.title("Dask Execution Time vs Number of Workers")
        plt.xlabel("Number of Workers")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "worker_scaling.png"), dpi=300)
        plt.close()  # Close the figure to prevent overlapping plots
    
    # Chunk size plot
    if "chunk_size" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="chunk_size", y="execution_time", marker="o")
        plt.title("Dask Execution Time vs Chunk Size")
        plt.xlabel("Chunk Size")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "chunk_size_scaling.png"), dpi=300)
        plt.close()  # Close the figure to prevent overlapping plots
    
    # Function performance plot
    if "function" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="N", y="execution_time", hue="function", marker="o")
        plt.title("Function Execution Time vs N")
        plt.xlabel("Number of Particles (N)")
        plt.ylabel("Execution Time (seconds)")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "function_performance.png"), dpi=300)
        plt.close()  # Close the figure to prevent overlapping plots
    
    # Save raw data
    df.to_csv(os.path.join(output_dir, f"dask_analysis_{int(time.time())}.csv"), index=False)

def main():
    # Create output directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Analyze worker scaling
    n_workers_list = [1, 2, 4, 8]
    worker_results = analyze_dask_scaling(n_workers_list, N=200, t_end=0.5)
    plot_scaling_results(worker_results)
    
    # Analyze chunk size impact
    chunk_sizes = [10, 25, 50, 100, 200]
    chunk_results = analyze_task_granularity(chunk_sizes, N=200, t_end=0.5)
    plot_scaling_results(chunk_results)
    
    # Analyze function performance
    N_values = [10, 50, 100, 200, 500]
    function_results = analyze_function_performance(N_values)
    plot_scaling_results(function_results)

if __name__ == "__main__":
    main() 