'''simulate.py'''
#!/usr/bin/env python3

#### IMPORTS ####
import os
import csv
import json
import argparse
import logging

import numpy as np
import openmm
from openmm.app import (
    Simulation, Modeller, ForceField, PDBFile, PDBxFile, DCDFile,
    StateDataReporter, PME, HBonds
)
from openmm.unit import femtoseconds, picosecond, nanometer, kelvin, atmospheres, kilojoules_per_mole
from pdbfixer.pdbfixer import PDBFixer

#### GLOBAL ####

logger = logging.getLogger(__name__)

# Residue names added by addSolvent (water + neutralizing/common ions). Atoms in
# these residues are stripped from the logged trajectories, which only the
# protein-only analyses (analyze.py) consume.
SOLVENT_RESIDUES = {'HOH', 'WAT', 'TIP3', 'TIP4', 'SPC',
                    'NA', 'CL', 'K', 'MG', 'ZN', 'CA', 'LI', 'CS', 'RB', 'BR'}


class SubsetDCDReporter:
    '''Like openmm.app.DCDReporter but writes only a subset of atoms each frame.

    subsetTopology must describe exactly the atoms in subsetIndices (in order),
    so a matching topology PDB can be loaded alongside the trajectory.'''

    def __init__(self, file, reportInterval, subsetTopology, subsetIndices,
                 enforcePeriodicBox=False, append=False):
        self._reportInterval = reportInterval
        self._topology = subsetTopology
        self._indices = subsetIndices
        self._enforcePeriodicBox = enforcePeriodicBox
        self._append = append
        # 'r+b' (not 'ab') on append: DCDFile must read the existing header and
        # rewrite its frame count, so it needs read+write on the same handle.
        self._out = open(file, 'r+b' if append else 'wb')
        self._dcd = None

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True)[self._indices, :]
        if self._dcd is None:
            self._dcd = DCDFile(self._out, self._topology,
                                simulation.integrator.getStepSize(),
                                simulation.currentStep, self._reportInterval,
                                append=self._append)
        self._dcd.writeModel(positions, periodicBoxVectors=state.getPeriodicBoxVectors())

    def __del__(self):
        self._out.close()


class StateXMLReporter:
    '''Periodically serialize the full simulation State to XML (saveState format)
    so production can be restarted in a fresh process on any GPU/host.

    The state is written to a temp file and atomically renamed over the target
    (os.replace is atomic on POSIX). This guarantees that a job killed mid-write
    never leaves a truncated state.xml -- a truncated state would be unparseable
    and destroy the restart point exactly when it is needed.

    saveState pulls the full state from the Context itself, so this reporter
    requests nothing from OpenMM's report state (all flags False).'''

    def __init__(self, file, reportInterval):
        self._reportInterval = reportInterval
        self._file = file

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        tmp = self._file + '.tmp'
        simulation.saveState(tmp)
        os.replace(tmp, self._file)

#### CLASS ####

class model:
    '''Modular OpenMM MD pipeline: prepare -> setup -> minimize -> equilibrate -> run

    All parameters are set at construction time, so the pipeline can be run as:
        sim = model(pdbFile=...)
        sim.prepare()
        sim.setup()
        sim.minimize()
        sim.equilibrate()
        sim.run()
    '''

    def __init__(self, directory,
                       pdbFile=None,
                       write=False,
                       mainForcefield='amber19-all.xml',
                       waterForcefield='amber19/tip3pfb.xml',
                       stepsize=2*femtoseconds,
                       padding=10.0*nanometer,
                       boxShape='octahedron',
                       temperature=310*kelvin,
                       pressure=1*atmospheres,
                       nonbondedMethod=PME,
                       nonbondedCutoff=1*nanometer,
                       constraints=HBonds,
                       friction=1/picosecond,
                       etimeNVT=100,
                       etimeNPT=100,
                       estepNVT=3.1*kelvin,
                       estepNPT=25,
                       plateauTolerance=5e-5,
                       sampleInterval=500,
                       smoothFraction=0.1,
                       maxEquilChunks=10,
                       steps=50000000,            # 100 ns at a 2 fs stepsize
                       logInterval=5000,          # text state log every 10 ps
                       interval=50000):           # trajectory frame AND state.xml every 100 ps
        # All run artifacts live under self.directory with fixed names. The
        # directory fully describes a run, so restart() needs only this path.
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        join = os.path.join
        self.params_json  = join(directory, 'params.json')    # restart manifest + record
        self.system_xml   = join(directory, 'system.xml')     # serialized production System
        self.solvated_pdb = join(directory, 'solvated.pdb')   # full (solvated) topology for restart
        self.state_xml    = join(directory, 'state.xml')      # rolling saveState (restart point)
        self.model_pdb    = join(directory, 'model.pdb')      # protein-only topology (analysis)
        self.trajectory   = join(directory, 'trajectory.dcd')   # production (protein-only)
        self.etrajectory  = join(directory, 'etrajectory.dcd')  # equilibration (protein-only)
        self.eenergy      = join(directory, 'eenergy.csv')      # NPT equilibration energy trace
        self.info         = join(directory, 'production.log')   # production state log
        self.einfo        = join(directory, 'equilibration.log')  # equilibration state log

        # prepare()
        self.pdbFile = pdbFile
        self.write = write

        # setup()
        self.mainForcefield = mainForcefield
        self.waterForcefield = waterForcefield
        self.stepsize = stepsize
        self.integrator = openmm.VerletIntegrator(self.stepsize)

        # minimize()
        self.padding = padding
        self.boxShape = boxShape

        # equilibrate()
        self.temperature = temperature
        self.pressure = pressure
        self.nonbondedMethod = nonbondedMethod
        self.nonbondedCutoff = nonbondedCutoff
        self.constraints = constraints
        self.friction = friction
        
        ## NVT
        self.etimeNVT = etimeNVT
        self.estepNVT = estepNVT
        
        ## NPT
        self.etimeNPT = etimeNPT
        self.estepNPT = estepNPT
        
        ## NPT loop
        self.plateauTolerance = plateauTolerance
        self.sampleInterval = sampleInterval
        self.smoothFraction = smoothFraction
        self.maxEquilChunks = maxEquilChunks

        # run()
        self.steps = steps
        self.logInterval = logInterval
        self.interval = interval

        # populated as the pipeline runs
        self.fixer = None
        self.file = None
        self.modeller = None
        self.forcefield = None
        self.system = None
        self.simulation = None
        self.subsetTopology = None   # protein-only topology, shared by both trajectories
        self.subsetIndices = None

    def prepare(self):
        '''Load and fix self.pdbFile (missing residues/atoms/hydrogens at pH 7).
        Stores self.fixer always; writes the fixed structure to disk only if self.write=True.'''
        base, ext = os.path.splitext(self.pdbFile)
        ext = ext.lower()
        if ext not in ('.pdb', '.cif'):
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info(f"Fixing {self.pdbFile}..")
        fixer = PDBFixer(filename=self.pdbFile)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        self.fixer = fixer
        self.file = os.path.join(self.directory, os.path.basename(base) + f'_fixed{ext}')

        if self.write:
            writer = PDBxFile if ext == '.cif' else PDBFile
            with open(self.file, 'w') as fh:
                writer.writeFile(fixer.topology, fixer.positions, fh)
            logger.info(f"Wrote {self.file}")

        logger.info(f"Fixed {self.pdbFile}. Missing residues: {fixer.missingResidues}")

    def setup(self):
        '''Build modeller, forcefield, and minimization-ready system from self.fixer'''
        self.modeller = Modeller(self.fixer.topology, self.fixer.positions)
        self.forcefield = ForceField(self.mainForcefield, self.waterForcefield)
        self.system = self.forcefield.createSystem(self.modeller.topology)

    def minimize(self):
        '''Minimize the solute, then add a water box around the minimized positions'''
        logger.info("Minimizing solute..")
        minimizer = Simulation(self.modeller.topology, self.system, self.integrator)
        minimizer.context.setPositions(self.modeller.positions)
        minimizer.minimizeEnergy()
        self.modeller.positions = minimizer.context.getState(getPositions=True).getPositions()
        logger.info("Solute minimized")

        logger.info("Adding water box..")
        self.modeller.addSolvent(self.forcefield, padding=self.padding, boxShape=self.boxShape)
        logger.info("Water box added")

    def equilibrate(self):
        '''Build the production system (nonbondedMethod, constraints, barostat), minimize the
        solvated system, then equilibrate in two phases: NVT warm-up to
        self.temperature, followed by NPT with the barostat enabled.'''
        self.system = self.forcefield.createSystem(self.modeller.topology,
                                                     nonbondedMethod=self.nonbondedMethod,
                                                     nonbondedCutoff=self.nonbondedCutoff,
                                                     constraints=self.constraints)

        self.barostat = openmm.MonteCarloBarostat(self.pressure, self.temperature, 0)
        self.system.addForce(self.barostat)

        self.integrator = openmm.LangevinMiddleIntegrator(self.temperature, self.friction, self.stepsize)
        self.simulation = Simulation(self.modeller.topology, self.system, self.integrator)
        self.simulation.context.setPositions(self.modeller.positions)

        logger.info("Minimizing solvated system..")
        self.simulation.minimizeEnergy()
        state = self.simulation.context.getState(getEnergy=True, getPositions=True)
        logger.info(f"Energy: {state.getPotentialEnergy()}")

        # Protein-only topology (solvent stripped), shared by both stripped
        # trajectories. Written once as model.pdb -- the topology for analysis.
        self.subsetTopology, self.subsetIndices = self._proteinSubset()
        subsetPositions = state.getPositions(asNumpy=True)[self.subsetIndices, :]
        with open(self.model_pdb, 'w') as fh:
            PDBFile.writeFile(self.subsetTopology, subsetPositions, fh)

        # Persist the restart artifacts now that the production system exists:
        # the serialized System (forces + barostat), the FULL solvated topology
        # (restart needs every atom, unlike the stripped model.pdb), and the run
        # manifest. With the rolling state.xml written during production, these
        # let restart() rebuild the Simulation in a fresh process on any GPU/host.
        with open(self.system_xml, 'w') as fh:
            fh.write(openmm.XmlSerializer.serialize(self.system))
        with open(self.solvated_pdb, 'w') as fh:
            PDBFile.writeFile(self.modeller.topology, state.getPositions(asNumpy=True), fh)
        self._writeParams()

        if self.etimeNVT == 0 and self.etimeNPT == 0:
            return

        self.simulation.reporters.append(
            StateDataReporter(self.einfo, reportInterval=50, step=True, temperature=True,
                              volume=True, potentialEnergy=True, speed=True))
        self.simulation.reporters.append(
            SubsetDCDReporter(self.etrajectory, 500, self.subsetTopology, self.subsetIndices))

        stepsizeFs = self.stepsize.value_in_unit(femtoseconds)

        if self.etimeNVT > 0:
            logger.info("First equilibration (NVT warm-up)..")
            nIncrements = round(self.temperature.value_in_unit(kelvin) / self.estepNVT.value_in_unit(kelvin))
            totalStepsNVT = round(self.etimeNVT * 1000 / stepsizeFs)
            tsteps = totalStepsNVT // nIncrements
            T = self.estepNVT
            for i in range(nIncrements):
                self.integrator.setTemperature(T)
                self.simulation.step(tsteps)
                T += self.estepNVT

        if self.etimeNPT > 0:
            # NPT runs in chunks until the smoothed potential energy plateaus
            # (relative drift < plateauTolerance). The full trace is written to
            # self.eenergy. RECOMMENDED: inspect that equilibration curve before
            # launching production -- automated plateau detection is a heuristic,
            # not a guarantee. Stiff or slowly-relaxing systems may need a larger
            # plateauTolerance window (smaller value) or a longer etimeNPT per
            # chunk to avoid declaring convergence while still drifting.
            logger.info("Second equilibration (NPT, barostat enabled)..")
            self.barostat.setFrequency(self.estepNPT)

            chunkSteps = round(self.etimeNPT * 1000 / stepsizeFs)
            nSamples = max(chunkSteps // self.sampleInterval, 2)
            sampleTimePs = self.sampleInterval * stepsizeFs / 1000  # ps between energy samples
            window = max(1, min(round(self.smoothFraction * nSamples), nSamples))
            kernel = np.ones(window) / window
            offset = (window - 1) // 2  # centering offset of the 'valid' moving average

            with open(self.eenergy, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time_ps', 'potential_energy_kJ_mol', 'smoothed_kJ_mol', 'chunk'])

                elapsedPs = 0.0
                converged = False
                for chunk in range(self.maxEquilChunks):
                    # Sample the potential energy across one chunk of NPT.
                    times, energies = [], []
                    for _ in range(nSamples):
                        self.simulation.step(self.sampleInterval)
                        elapsedPs += sampleTimePs
                        e = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
                        times.append(elapsedPs)
                        energies.append(e.value_in_unit(kilojoules_per_mole))

                    # Boxcar-smooth the noisy trace (centered 'valid' moving average).
                    smoothed = np.convolve(energies, kernel, mode='valid')
                    smoothedTimes = times[offset:offset + len(smoothed)]

                    # Align smoothed values back onto raw samples for the CSV (edges blank).
                    smoothedColumn = [None] * len(energies)
                    for i, s in enumerate(smoothed):
                        smoothedColumn[i + offset] = s
                    for t, e, s in zip(times, energies, smoothedColumn):
                        writer.writerow([f'{t:.4f}', f'{e:.4f}',
                                         '' if s is None else f'{s:.4f}', chunk])
                    csvfile.flush()

                    # Plateau test: slope of the smoothed trace, normalized by its
                    # magnitude -> fractional energy drift per ps (system-size independent).
                    slope = np.polyfit(smoothedTimes, smoothed, 1)[0]  # kJ/mol per ps
                    relativeDrift = abs(slope) / abs(np.mean(smoothed))  # 1/ps
                    logger.info(f"NPT chunk {chunk}: mean PE = {np.mean(smoothed):.2f} kJ/mol, "
                                f"relative drift = {relativeDrift:.2e} /ps")

                    if relativeDrift < self.plateauTolerance:
                        logger.info("NPT equilibration converged (energy plateaued)")
                        converged = True
                        break

                if not converged:
                    logger.warning(f"NPT equilibration did not plateau within {self.maxEquilChunks} chunks")

    def _proteinSubset(self):
        '''Topology + original-topology atom indices for the non-solvent
        (protein/ligand) atoms, used to strip water/ions from the trajectory.'''
        modeller = Modeller(self.modeller.topology, self.modeller.positions)
        solvent = [a for a in modeller.topology.atoms() if a.residue.name in SOLVENT_RESIDUES]
        modeller.delete(solvent)
        indices = [a.index for a in self.modeller.topology.atoms()
                   if a.residue.name not in SOLVENT_RESIDUES]
        return modeller.topology, indices

    def _writeParams(self):
        '''Write the restart manifest / run record. Holds everything restart()
        needs that is NOT already captured by system.xml (the integrator config)
        or solvated.pdb (the topology): integrator parameters and run config.'''
        params = {
            'pdbFile': self.pdbFile,
            'stepsize_fs': self.stepsize.value_in_unit(femtoseconds),
            'temperature_K': self.temperature.value_in_unit(kelvin),
            'friction_per_ps': self.friction.value_in_unit(picosecond**-1),
            'pressure_atm': self.pressure.value_in_unit(atmospheres),
            'steps': self.steps,
            'interval': self.interval,
            'logInterval': self.logInterval,
        }
        with open(self.params_json, 'w') as fh:
            json.dump(params, fh, indent=2)

    def _resume(self, targetSteps, interval, logInterval, append):
        '''Attach production reporters and step until currentStep == targetSteps.

        Reporter order is deliberate: the trajectory writer runs BEFORE the
        state writer, so a crash between them re-runs (at most) the last frame on
        restart rather than dropping it -- a duplicate frame is preferred over a
        gap. Both fire every `interval` steps so the trajectory and state.xml
        stay in lockstep (their restart points can never diverge).'''
        self.simulation.reporters = []
        self.simulation.reporters.append(
            StateDataReporter(self.info, reportInterval=logInterval, step=True, temperature=True,
                              volume=True, potentialEnergy=True, speed=True, append=append))
        self.simulation.reporters.append(
            SubsetDCDReporter(self.trajectory, interval, self.subsetTopology,
                              self.subsetIndices, append=append))
        self.simulation.reporters.append(StateXMLReporter(self.state_xml, interval))

        remaining = targetSteps - self.simulation.currentStep
        if remaining <= 0:
            logger.info("Target step count already reached; nothing to run")
            return
        logger.info(f"Running production ({remaining} steps remaining)..")
        self.simulation.step(remaining)

    def run(self):
        '''Run the production simulation for self.steps timesteps.

        The simulation itself runs in explicit solvent (water + ions); only the
        logging is solvent-stripped. The solvent is omitted from model.pdb and
        trajectory.dcd to save space -- every downstream analysis is protein-only.
        model.pdb (the trajectory's topology) is written in equilibrate().'''
        self.simulation.currentStep = 0
        # Reset the clock so state.xml's elapsed time measures PRODUCTION progress
        # (restart() derives the resume step from it; equilibration time must not
        # leak in).
        self.simulation.context.setTime(0 * picosecond)
        # Write an initial restart point so even a crash within the first
        # `interval` can resume from production start.
        self.simulation.saveState(self.state_xml)

        self._resume(self.steps, self.interval, self.logInterval, append=False)

    def _load(self):
        '''Rebuild the Simulation in a fresh process from the persisted run dir
        and load the last saved state.xml. Shared by restart() and extend().
        Returns the parsed params.json manifest.

        Resumes PRODUCTION ONLY: if state.xml is absent, equilibration never
        finished -- resubmit the initial job for this design (equilibration is
        cheap relative to production).'''
        if not os.path.exists(self.state_xml):
            raise FileNotFoundError(
                f"No {self.state_xml}: production never started (equilibration "
                f"did not finish). Resubmit the initial job for this design.")

        with open(self.params_json) as fh:
            p = json.load(fh)

        pdb = PDBFile(self.solvated_pdb)
        with open(self.system_xml) as fh:
            self.system = openmm.XmlSerializer.deserialize(fh.read())

        integrator = openmm.LangevinMiddleIntegrator(
            p['temperature_K'] * kelvin,
            p['friction_per_ps'] / picosecond,
            p['stepsize_fs'] * femtoseconds)
        self.simulation = Simulation(pdb.topology, self.system, integrator)
        self.simulation.loadState(self.state_xml)

        # loadState restores the context clock but NOT simulation.currentStep, so
        # derive production progress from the elapsed time (reset to 0 in run()).
        elapsedPs = self.simulation.context.getState().getTime().value_in_unit(picosecond)
        self.simulation.currentStep = round(elapsedPs * 1000 / p['stepsize_fs'])

        # Re-derive the protein-only subset for the appending trajectory writer.
        self.modeller = Modeller(pdb.topology, pdb.positions)
        self.subsetTopology, self.subsetIndices = self._proteinSubset()
        return p

    def restart(self):
        '''Resume production in self.directory toward its existing step target,
        from the last saved state.xml (saveState format -- portable GPU/host).'''
        p = self._load()
        logger.info(f"Restarting at step {self.simulation.currentStep}; target {p['steps']}")
        self._resume(p['steps'], p['interval'], p['logInterval'], append=True)

    def extend(self):
        '''Continue a run toward a NEW, larger total step target (self.steps),
        e.g. 100 ns -> 150 ns. The new target is persisted to params.json so
        subsequent restarts resume toward it. Same resume machinery as restart;
        only the target (and the manifest) change.'''
        p = self._load()
        if self.steps <= p['steps']:
            raise ValueError(
                f"--extend target {self.steps} must exceed the current target "
                f"{p['steps']}; pass a larger --steps")

        p['steps'] = self.steps
        with open(self.params_json, 'w') as fh:
            json.dump(p, fh, indent=2)

        logger.info(f"Extending from step {self.simulation.currentStep} to {self.steps}")
        self._resume(self.steps, p['interval'], p['logInterval'], append=True)


#### CLI ####

def buildParser():
    parser = argparse.ArgumentParser(description='Simulate a PDB/CIF using OpenMM')
    parser.add_argument("--dir", required=True, help="Run directory (all artifacts written here)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--restart", action="store_true",
                      help="Resume production from --dir toward its existing --steps target")
    mode.add_argument("--extend", action="store_true",
                      help="Resume production from --dir toward a NEW larger --steps target")
    parser.add_argument("--pdb", help="Input PDB or CIF file (receptor+ligand); required unless --restart/--extend")
    parser.add_argument("--write_fixed", action="store_true", help="Write the fixed structure to disk")
    parser.add_argument("--padding", type=float, default=10.0, help="Water box padding in nm")
    parser.add_argument("--box_shape", type=str, default="octahedron", help="Water box shape")
    parser.add_argument("--temperature", type=float, default=310, help="Temperature for simulation in Kelvin")
    parser.add_argument("--etimeNVT", type=int, default=100, help="Picoseconds spent in NVT warm-up")
    parser.add_argument("--etimeNPT", type=int, default=100, help="Picoseconds spent per NPT equilibration chunk")
    parser.add_argument("--estepNVT", type=float, default=3.1, help="Temperature increment (K) per NVT warm-up step")
    parser.add_argument("--estepNPT", type=int, default=25, help="Barostat frequency (steps) during NPT")
    parser.add_argument("--plateau_tolerance", type=float, default=5e-5, help="NPT plateau tolerance: max fractional energy drift per ps")
    parser.add_argument("--sample_interval", type=int, default=500, help="Steps between NPT energy samples")
    parser.add_argument("--smooth_fraction", type=float, default=0.1, help="Moving-average window as a fraction of samples per chunk")
    parser.add_argument("--max_equil_chunks", type=int, default=10, help="Max number of NPT equilibration chunks")
    parser.add_argument("--steps", type=int, default=50000000, help="Number of 2fs production time steps (default 100 ns)")
    parser.add_argument("--interval", type=int, default=50000, help="Steps between trajectory frames AND state.xml saves (default 100 ps)")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = buildParser()
    args = parser.parse_args()

    if args.restart:
        sim = model(directory=args.dir)
        sim.restart()
    elif args.extend:
        sim = model(directory=args.dir, steps=args.steps)
        sim.extend()
    else:
        if not args.pdb:
            parser.error("--pdb is required unless --restart/--extend is set")

        sim = model(directory=args.dir,
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

        if sim.etimeNVT > 0 or sim.etimeNPT > 0:
            sim.run()
