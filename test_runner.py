import unittest
import random

import numpy as np
import torch
import dask.array as da

# Import the main function from the module containing your simulation.
# Replace 'nbody_simulation' with the actual module name.
from nbody_original.nbody_original import main as original_main
from nbody_cython.nbody_cython import main as cython_main
from nbody_original.nbody_pure_python import main as pure_main
from nbody_pytorch.nbody_pytorch import main as torch_main
from nbody_dask.nbody_dask import main as dask_main


class TestNBodySimulation(unittest.TestCase):
    def get_sim_params(self):
        return {
            "N": 100,
            "t": 0,
            "t_end": 1.0,
            "dt": 0.01,
            "softening": 0.1,
            "G": 1.0,
            "plot_real_time": False,
        }

    def test_cython_implementation(self):
        original_pos, original_vel, original_KE, original_PE = original_main(**self.get_sim_params())
        cython_pos, cython_vel, cython_KE, cython_PE = cython_main(**self.get_sim_params())
        np.testing.assert_array_almost_equal(original_pos, cython_pos)
        np.testing.assert_array_almost_equal(original_vel, cython_vel)
        np.testing.assert_array_almost_equal(original_KE, cython_KE)
        np.testing.assert_array_almost_equal(original_PE, cython_PE)

    def test_pure_implementations(self):
        N = 100
        random.seed(42)
        pos = [[random.gauss(0, 1) for _ in range(3)] for _ in range(N)]
        vel = [[random.gauss(0, 1) for _ in range(3)] for _ in range(N)]
        original_pos, original_vel, original_KE, original_PE = original_main(
            **{
                **self.get_sim_params(),
                "pos": np.asarray(pos),
                "vel": np.asarray(vel),
            }
        )
        pure_pos, pure_vel, pure_KE, pure_PE = pure_main(
            **{
                **self.get_sim_params(),
                "pos": pos,
                "vel": vel,
            }
        )
        np.testing.assert_array_almost_equal(original_pos, pure_pos)
        np.testing.assert_array_almost_equal(original_vel, pure_vel)
        np.testing.assert_array_almost_equal(original_KE, pure_KE)
        np.testing.assert_array_almost_equal(original_PE, pure_PE)

    def test_pytorch_implmentation(self):
        np.random.seed(42)
        N = 100
        pos = np.random.randn(N, 3).astype(np.float32)
        vel = np.random.randn(N, 3).astype(np.float32)
        original_pos, original_vel, original_KE, original_PE = original_main(
            **{
                **self.get_sim_params(),
                "pos": pos.copy(),
                "vel": vel.copy(),
            }
        )
        torch_pos, torch_vel, torch_KE, torch_PE = torch_main(
            **{
                **self.get_sim_params(),
                "pos_init": torch.from_numpy(pos.copy()),
                "vel_init": torch.from_numpy(vel.copy()),
            }
        )
        d = 5
        np.testing.assert_array_almost_equal(original_pos, torch_pos)
        np.testing.assert_array_almost_equal(original_vel, torch_vel, decimal=d)
        np.testing.assert_array_almost_equal(original_KE, torch_KE)
        np.testing.assert_array_almost_equal(original_PE, torch_PE, decimal=d)

    def test_dask_implementation(self):
        np.random.seed(42)
        N = 100
        pos = np.random.randn(N, 3).astype(np.float32)
        vel = np.random.randn(N, 3).astype(np.float32)
        original_pos, original_vel, original_KE, original_PE = original_main(
            **{
                **self.get_sim_params(),
                "dt": 1.0,
                # "N": N,
                # "pos": pos.copy(),
                # "vel": vel.copy(),
            }
        )
        dask_pos, dask_vel, dask_KE, dask_PE = dask_main(
            **{
                **self.get_sim_params(),
                # "N": N,
                # "pos": da.from_array(pos.copy()),
                # "vel": da.from_array(vel.copy()),
            }
        )
        d = 3
        np.testing.assert_array_almost_equal(original_pos, dask_pos, decimal=d)
        np.testing.assert_array_almost_equal(original_vel, dask_vel, decimal=d)
        np.testing.assert_array_almost_equal(original_KE[-1], dask_KE, decimal=d)
        np.testing.assert_array_almost_equal(original_PE[-1], dask_PE, decimal=1)

if __name__ == "__main__":
    unittest.main()
