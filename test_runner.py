import unittest
import numpy as np

# Import the main function from the module containing your simulation.
# Replace 'nbody_simulation' with the actual module name.
from nbody_original.nbody_original import main as original_main
from nbody_cython.nbody_cython import main as cython_main

class TestNBodySimulation(unittest.TestCase):

    def get_hparams(self):
        N = 5

        pos = np.array([
            [-1.38293429,  0.41895108,  0.24246185],
            [-0.64327503,  0.60574128, -1.77305099],
            [ 0.19334569, -0.98366222,  1.64624345],
            [ 0.26165781, -0.03424197, -1.36230998],
            [ 1.0503802,  -0.77989414,  0.36102601]
        ])

        vel = np.array([
            [ 0.22596995, -0.1907937,  -1.1076018 ],
            [ 1.41082299, -0.67049779, -0.15500014],
            [ 0.55513381, -0.99826169, -0.08625607],
            [-1.44084409, -0.14730157, -0.41200918],
            [-0.31469862,  0.2655889,   0.18458102]
        ])

        expected_pos = np.array([
            [-2.40688048,  2.04126908,  2.52167063],
            [ 6.73359867, -7.04393087, -10.46728271],
            [-2.4370289,   1.79196072,  2.2911926 ],
            [-0.41539033,  1.0836613,   2.56700758],
            [-1.99512458,  1.3539338,   2.20178223]
        ])

        expected_vel = np.array([
            [ 0.62921779,  0.72183399, -2.48757653],
            [ 0.54194208, -0.43518085, -0.8039434 ],
            [-1.19799469,  2.06863793,  1.80304629],
            [ 0.45737658,  0.02446882,  0.84070664],
            [-0.43054175, -2.37975989,  0.647767  ]
        ])

        return N, pos, vel, expected_pos, expected_vel

    def test_original_implementation(self):
        N, pos, vel, expected_pos, expected_vel = self.get_hparams()
        final_pos, final_vel = original_main(N=N, t=0, tEnd=10.0, dt=0.01, softening=0.1, G=1.0, plotRealTime=False, measureTime=False, pos=pos, vel=vel)

        np.testing.assert_array_almost_equal(final_pos, expected_pos, decimal=7)
        np.testing.assert_array_almost_equal(final_vel, expected_vel, decimal=7)

    def test_cython_implementation(self):
        N, pos, vel, expected_pos, expected_vel = self.get_hparams()
        final_pos, final_vel = cython_main(N=N, t=0, tEnd=10.0, dt=0.01, softening=0.1, G=1.0, plotRealTime=False, measureTime=False, pos=pos, vel=vel)
        
        np.testing.assert_array_almost_equal(final_pos, expected_pos, decimal=7)
        np.testing.assert_array_almost_equal(final_vel, expected_vel, decimal=7)
        
if __name__ == '__main__':
    unittest.main()
