from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-mode",
        choices=["original", "cython", "pytorch"],
        default="original",
        help="Choose execution mode: original (default), cython, or pytorch",
    )
    parser.add_argument("-N", type=int, help="Number of particles", default=100)
    parser.add_argument(
        "-t", type=int, help="Current time of the simulation", default=0
    )
    parser.add_argument(
        "-tEnd", type=float, help="Time at which simulation ends", default=10.0
    )
    parser.add_argument("-dt", type=float, help="Size of a timestep", default=0.01)
    parser.add_argument("-softening", type=float, help="Softening length", default=0.1)
    parser.add_argument("-G", type=float, help="Gravitational constant", default=1.0)
    parser.add_argument(
        "-plotRealTime",
        type=bool,
        help="Enable plotting as the simulation goes along",
        default=False,
    )
    args = parser.parse_args()

    match args.mode:
        case "cython":
            from nbody_cython.nbody_cython import main
        case "original":
            from nbody_original.nbody_original import main
        case _:
            from nbody_original.nbody_original import main

    kwargs = vars(args)
    del kwargs["mode"]
    main(**kwargs)


if __name__ == "__main__":
    main()
