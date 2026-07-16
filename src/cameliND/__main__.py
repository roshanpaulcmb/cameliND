'''cameliND command-line entry point: `python -m cameliND <verb>`.

    python -m cameliND simulate --pdb model.cif --dir run/
    python -m cameliND restart  --dir run/
    python -m cameliND extend   --dir run/ --steps 75000000

The `simulate` class lives in the internal `_simulate` submodule; this module is
the public CLI surface, so the run directory's `_simulate` path stays an
implementation detail.
'''
import logging
import argparse

from openmm.unit import nanometer, kelvin

from ._simulate import simulate


def buildParser():
    parser = argparse.ArgumentParser(prog="cameliND", description="cameliND OpenMM MD pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # simulate: fresh run from a structure
    run = sub.add_parser("simulate", help="Run a fresh simulation from a PDB/CIF")
    run.add_argument("--pdb", required=True, help="Input PDB or CIF file (receptor+ligand)")
    run.add_argument("--dir", required=True, help="Run directory (all artifacts written here)")
    run.add_argument("--write_fixed", action="store_true", help="Write the fixed structure to disk")
    run.add_argument("--padding", type=float, default=1.0, help="Water box padding in nm")
    run.add_argument("--box_shape", type=str, default="octahedron", help="Water box shape")
    run.add_argument("--temperature", type=float, default=310, help="Temperature for simulation in Kelvin")
    run.add_argument("--etimeNVT", type=int, default=100, help="Picoseconds spent in NVT warm-up")
    run.add_argument("--etimeNPT", type=int, default=100, help="Picoseconds spent per NPT equilibration chunk")
    run.add_argument("--estepNVT", type=float, default=3.1, help="Temperature increment (K) per NVT warm-up step")
    run.add_argument("--estepNPT", type=int, default=25, help="Barostat frequency (steps) during NPT")
    run.add_argument("--plateau_tolerance", type=float, default=5e-5, help="NPT plateau tolerance: max fractional energy drift per ps")
    run.add_argument("--sample_interval", type=int, default=500, help="Steps between NPT energy samples")
    run.add_argument("--smooth_fraction", type=float, default=0.1, help="Moving-average window as a fraction of samples per chunk")
    run.add_argument("--max_equil_chunks", type=int, default=10, help="Max number of NPT equilibration chunks")
    run.add_argument("--steps", type=int, default=50000000, help="Number of 2fs production time steps (default 100 ns)")
    run.add_argument("--interval", type=int, default=50000, help="Steps between trajectory frames AND state.xml saves (default 100 ps)")

    # restart: resume toward the existing target
    res = sub.add_parser("restart", help="Resume production toward its existing --steps target")
    res.add_argument("--dir", required=True, help="Run directory to resume")

    # extend: resume toward a NEW larger target
    ext = sub.add_parser("extend", help="Resume production toward a NEW larger --steps target")
    ext.add_argument("--dir", required=True, help="Run directory to extend")
    ext.add_argument("--steps", type=int, required=True, help="New total 2fs step target (must exceed the current target)")

    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = buildParser().parse_args(argv)

    if args.command == "restart":
        simulate(directory=args.dir).restart()
        return

    if args.command == "extend":
        simulate(directory=args.dir, steps=args.steps).extend()
        return

    # simulate
    sim = simulate(directory=args.dir,
                   pdbFile=args.pdb,
                   write=args.write_fixed,
                   padding=args.padding*nanometer,
                   boxShape=args.box_shape,
                   temperature=args.temperature*kelvin,
                   etimeNVT=args.etimeNVT,
                   etimeNPT=args.etimeNPT,
                   estepNVT=args.estepNVT*kelvin,
                   estepNPT=args.estepNPT,
                   plateauTolerance=args.plateau_tolerance,
                   sampleInterval=args.sample_interval,
                   smoothFraction=args.smooth_fraction,
                   maxEquilChunks=args.max_equil_chunks,
                   steps=args.steps,
                   interval=args.interval)

    sim.prepare()
    sim.setup()
    sim.minimize()
    sim.equilibrate()

    if sim.EtimeNVT > 0 or sim.EtimeNPT > 0:
        sim.run()


if __name__ == "__main__":
    main()
