import argparse
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-mode",
        choices=["original", "cython", "dask", "pytorch"],
        default="original",
        help="Choose execution mode: original (default), cython, dask, or pytorch",
    )
    parser.add_argument("-N", type=int, help="Number of particles", default=100)
    parser.add_argument(
        "-t", type=int, help="Current time of the simulation", default=0
    )
    parser.add_argument(
        "-t_end", type=float, help="Time at which simulation ends", default=10.0
    )
    parser.add_argument("-dt", type=float, help="Size of a timestep", default=0.01)
    parser.add_argument("-softening", type=float, help="Softening length", default=0.1)
    parser.add_argument("-G", type=float, help="Gravitational constant", default=1.0)
    parser.add_argument(
        "--plot_real_time",
        type=bool,
        help="Enable plotting as the simulation goes along",
        default=False,
    )
    parser.add_argument(
        "--measure_time",
        type=bool,
        help="Measure time",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    match args.mode:
        case "cython":
            from nbody_cython.nbody_cython import main as nbody
        case "original":
            from nbody_original.nbody_original import main as nbody
        case "dask":
            from nbody_dask.nbody_dask import main as nbody
        case "pytorch":
            from nbody_pytorch.nbody_pytorch import main as nbody
        case _:
            from nbody_original.nbody_original import main as nbody

    kwargs = vars(args)
    del kwargs["mode"]
    nbody(**kwargs)


if __name__ == "__main__":
    main()
