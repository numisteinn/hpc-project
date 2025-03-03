import unittest
import numpy as np

# Import the main function from the module containing your simulation.
# Replace 'nbody_simulation' with the actual module name.
from nbody_original.nbody_original import main as original_main
from nbody_cython.nbody_cython import main as cython_main


class TestNBodySimulation(unittest.TestCase):
    def get_sim_params(self):
        return {
            "N": 100,
            "t": 0,
            "t_end": 1.0,
            "dt": 0.01,
            "softening": 0.1,
            "G": 1.0,
        }

    def test_cython_implementation(self):
        original_pos, original_vel, original_KE, original_PE = original_main(
            **self.get_sim_params()
        )
        cython_pos, cython_vel, cython_KE, cython_PE = cython_main(
            **self.get_sim_params()
        )
        np.testing.assert_array_almost_equal(original_pos, cython_pos)
        np.testing.assert_array_almost_equal(original_vel, cython_vel)
        np.testing.assert_array_almost_equal(original_KE, cython_KE)
        np.testing.assert_array_almost_equal(original_PE, cython_PE)


if __name__ == "__main__":
    unittest.main()
