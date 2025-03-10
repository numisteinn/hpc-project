import os
import time
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
import torch

# Import implementations
from nbody_original.nbody_original import main as original_main

from nbody_cython.nbody_cython import main as cython_main
from nbody_original.nbody_pure_python import main as pure_main
from nbody_pytorch.nbody_pytorch import main as torch_main
from nbody_dask.nbody_dask import main as dask_main
from memory_profiler import memory_usage
from collections import namedtuple

Metrics = namedtuple(
    "Metrics", ["implementation", "N", "execution_time", "memory_usage"]
)


def run_benchmark(
    implementation,
    name,
    N,
    t_end=1.0,
    dt=0.01,
    softening=0.1,
    G=1.0,
    pos=None,
    vel=None,
    dask_pos=None,
    dask_vel=None,
    torch_pos=None,
    torch_vel=None,
    measurement="time",
):
    """Run a single benchmark and return metrics"""
    # Force garbage collection before each run
    gc.collect()

    # Set up parameters
    params = {
        "N": N,
        "t": 0,
        "t_end": t_end,
        "dt": dt,
        "softening": softening,
        "G": G,
        "plot_real_time": False,
        "measure_time": False,
    }

    # Add position and velocity based on implementation
    if name == "dask" and dask_pos is not None:
        params["pos"] = dask_pos
        params["vel"] = dask_vel
    elif name == "pytorch" and torch_pos is not None:
        params["pos_init"] = torch_pos
        params["vel_init"] = torch_vel
    elif pos is not None:
        params["pos"] = pos
        params["vel"] = vel

    # Initialize metrics
    metrics = Metrics(implementation=name, N=N, execution_time=None, memory_usage=None)

    # Measure time before if required
    if measurement == "time":
        # Run the implementation
        start_time = time.time()
        # Measure time after if required
        implementation(**params)
        end_time = time.time()
        metrics = metrics._replace(execution_time=end_time - start_time)
        print(f"Execution time measured: {metrics.execution_time} seconds.")
    else:
        # Measure memory after if required
        print("Measuring memory usage...")
        # List of memory usage values
        multiprocess = False
        timeout = 2 * 60  # 2 minutesk
        m = memory_usage(
            (implementation, (), params),
            interval=0.1,
            timeout=timeout,
            multiprocess=multiprocess,
        )
        metrics = metrics._replace(memory_usage=m)
        print(f"Memory usage measured: {metrics.memory_usage} MB.")

    print(f"Benchmark completed for {name} with N={N}.")
    return metrics


def benchmark_all_implementations(
    N_values,
    runs_per_N=3,
    variations=None,
    measurement="time",  # or "memory"
):
    """Benchmark all implementations for different N values"""
    results = []
    variations = variations or ["pure", "original", "cython", "pytorch", "dask"]
    for N in N_values:
        print(f"Benchmarking with N={N}")

        # Generate consistent initial conditions for all implementations
        np.random.seed(42)
        pos_np = np.random.randn(N, 3).astype(np.float32)
        vel_np = np.random.randn(N, 3).astype(np.float32)

        # Create Dask arrays
        dask_pos = da.from_array(pos_np.copy(), chunks=(N // 4, 3))
        dask_vel = da.from_array(vel_np.copy(), chunks=(N // 4, 3))

        # Create PyTorch tensors
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("mps")
        torch_pos = torch.tensor(pos_np.copy(), device=device)
        torch_vel = torch.tensor(vel_np.copy(), device=device)

        for run in range(runs_per_N):
            print(f"Run {run + 1}/{runs_per_N}")

            # Skip pure Python for large N to avoid excessive runtime
            if "python" in variations and (
                N <= 500 or run == 0
            ):  # Only run pure Python once for large N
                try:
                    print("Running pure Python implementation")
                    results.append(
                        run_benchmark(
                            pure_main,
                            "pure",
                            N,
                            pos=pos_np.copy(),
                            vel=vel_np.copy(),
                            measurement=measurement,
                        )
                    )
                except Exception as e:
                    print(f"Error in pure Python implementation: {e}")

            if "original" in variations:
                try:
                    print("Running original NumPy implementation")
                    results.append(
                        run_benchmark(
                            original_main,
                            "original",
                            N,
                            pos=pos_np.copy(),
                            vel=vel_np.copy(),
                            measurement=measurement,
                        )
                    )
                except Exception as e:
                    print(f"Error in original implementation: {e}")

            if "cython" in variations:
                try:
                    print("Running Cython implementation")
                    results.append(
                        run_benchmark(
                            cython_main,
                            "cython",
                            N,
                            pos=pos_np.copy().astype(np.float64),
                            vel=vel_np.copy().astype(np.float64),
                            measurement=measurement,
                        )
                    )
                except Exception as e:
                    print(f"Error in Cython implementation: {e}")

            if "pytorch" in variations:
                try:
                    print("Running PyTorch implementation")
                    results.append(
                        run_benchmark(
                            torch_main,
                            "pytorch",
                            N,
                            torch_pos=torch_pos.clone(),
                            torch_vel=torch_vel.clone(),
                            measurement=measurement,
                        )
                    )
                except Exception as e:
                    print(f"Error in PyTorch implementation: {e}")

            if "dask" in variations:
                try:
                    print("Running Dask implementation")
                    results.append(
                        run_benchmark(
                            dask_main,
                            "dask",
                            N,
                            pos=dask_pos.copy(),
                            vel=dask_vel.copy(),
                            measurement=measurement,
                        )
                    )
                except Exception as e:
                    print(f"Error in Dask implementation: {e}")

    return pd.DataFrame(results)


def plot_results(df, speedup_df, output_dir="benchmark_results"):
    """Generate plots from benchmark results"""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set(style="whitegrid")

    # Execution time by N and implementation
    # Ignore if execution_time is not in df
    if "execution_time" in df.columns:
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=df, x="N", y="execution_time", hue="implementation", marker="o"
        )
        plt.title("Execution Time by Number of Particles")
        plt.xlabel("Number of Particles (N)")
        plt.ylabel("Execution Time (seconds)")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "execution_time.png"), dpi=300)
        plt.close()  # Close the figure to prevent overlapping plots

    if "memory_usage" in df.columns:
        # Memory usage by N and implementation
        plt.figure(figsize=(12, 8))

        # If memory_usage is vector of lists, convert to mean
        if isinstance(df["memory_usage"].values[0], list):
            df["memory_usage"] = df["memory_usage"].apply(np.mean)

        sns.lineplot(data=df, x="N", y="memory_usage", hue="implementation", marker="o")
        plt.title("Memory Usage by Number of Particles")
        plt.xlabel("Number of Particles (N)")
        plt.ylabel("Memory Usage (MB)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "memory_usage.png"), dpi=300)
        plt.close()  # Close the figure to prevent overlapping plots

    if speedup_df is not None:
        # Speedup relative to original implementation
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=speedup_df, x="N", y="speedup", hue="implementation", marker="o"
        )
        plt.axhline(y=1, color="r", linestyle="--")
        plt.title("Speedup Relative to Original Implementation")
        plt.xlabel("Number of Particles (N)")
        plt.ylabel("Speedup Factor (higher is better)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "speedup.png"), dpi=300)
        plt.close()  # Close the figure to prevent overlapping plots


def get_speedup(df):
    pivot_df = df.pivot_table(
        index=["N", "implementation"], values="execution_time"
    ).reset_index()

    # Calculate speedup for each N
    speedups = []
    for n in pivot_df["N"].unique():
        n_df = pivot_df[pivot_df["N"] == n]
        original_time = n_df[n_df["implementation"] == "original"][
            "execution_time"
        ].values[0]

        for _, row in n_df.iterrows():
            speedups.append(
                {
                    "N": n,
                    "implementation": row["implementation"],
                    "speedup": original_time / row["execution_time"],
                }
            )

    return pd.DataFrame(speedups)


def print_summary(results):
    # Print summary
    print("\nBenchmark Summary:")
    summary = (
        results.groupby(["implementation", "N"])["execution_time"].mean().reset_index()
    )
    print(summary)


def main():
    output_dir = "benchmark_results"
    if "--from_csv" in sys.argv:
        print("Plotting results from existing CSV file")
        results = pd.read_csv("benchmark_results/benchmark_results.csv")
        speedup_df = get_speedup(results)
        plot_results(results, speedup_df, output_dir=output_dir)
        print_summary(results)
        return

    # For quick testing, use a smaller set
    is_quick = "--quick" in sys.argv
    if is_quick:
        print("Running quick benchmark with fewer N values")
        N_values = [10, 100, 500]
        runs_per_N = 3
    else:
        print("Running full benchmark with all N values")
        # Define N values to test
        N_values = [10, 50, 100, 500, 1000, 2000]
        runs_per_N = 3

    # Run benchmarks
    variations = None
    print("Saving results to CSV")
    measurement = "memory"
    results = benchmark_all_implementations(
        N_values, runs_per_N, variations=variations, measurement=measurement
    )
    suffix = "_quick" if is_quick else ""
    results.to_csv(
        os.path.join(output_dir, f"benchmark_results{suffix}.csv"), index=False
    )
    speedup_df = None
    if measurement == "time":
        speedup_df = get_speedup(results)
        # Save raw data
        speedup_df.to_csv(
            os.path.join(output_dir, f"speedup_results{suffix}.csv"), index=False
        )

        print_summary(results)

    # Plot results
    plot_results(results, speedup_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
