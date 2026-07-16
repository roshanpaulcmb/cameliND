'''simulate.py'''
#!/usr/bin/env python3

#### IMPORTS ####
import os
import csv
import json
import logging

import numpy as np
import openmm
from openmm.app import (
    Simulation, Modeller, ForceField, PDBFile, PDBxFile,
    StateDataReporter, PME, HBonds
)
from openmm.unit import femtoseconds, picosecond, nanometer, kelvin, atmospheres, kilojoules_per_mole
from pdbfixer.pdbfixer import PDBFixer

from ._report import SubsetDCDReporter, StateXMLReporter

#### GLOBAL ####

logger = logging.getLogger(__name__)

# Residue names added by addSolvent (water + neutralizing/common ions). Atoms in
# these residues are stripped from the logged trajectories, which only the
# protein-only analyses (analyze.py) consume.
SOLVENT_RESIDUES = {'HOH', 'WAT', 'TIP3', 'TIP4', 'SPC',
                    'NA', 'CL', 'K', 'MG', 'ZN', 'CA', 'LI', 'CS', 'RB', 'BR'}


def _cudaPlatform():
    '''Return the CUDA platform, raising if it is unavailable.

    Pinned explicitly: left to auto-select, OpenMM silently falls back to the CPU
    platform when a CUDA context cannot be initialized, which turns a dead GPU into
    a run that crawls for hours and then dies mid-equilibration instead of failing
    here. This protocol is GPU-only, so no CUDA is an error, not a slow path.'''
    platform = openmm.Platform.getPlatformByName('CUDA')
    logger.info(f"Using OpenMM platform: {platform.getName()}")
    return platform

#### CLASS ####

class simulate:
    '''Modular OpenMM MD pipeline: prepare -> setup -> minimize -> equilibrate -> run

    All parameters are set at construction time, so the pipeline can be run as:
        sim = simulate(pdbFile=...)
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
                       padding=1.0*nanometer,
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
        # All run artifacts live under self.Directory with fixed names. The
        # directory fully describes a run, so restart() needs only this path.
        self.Directory = directory
        os.makedirs(directory, exist_ok=True)
        join = os.path.join
        self.ParamsJson  = join(directory, 'params.json')    # restart manifest + record
        self.SystemXml   = join(directory, 'system.xml')     # serialized production System
        self.SolvatedPdb = join(directory, 'solvated.pdb')   # full (solvated) topology for restart
        self.StateXml    = join(directory, 'state.xml')      # rolling saveState (restart point)
        self.ModelPdb    = join(directory, 'model.pdb')      # protein-only topology (analysis)
        self.Trajectory   = join(directory, 'trajectory.dcd')   # production (protein-only)
        self.Etrajectory  = join(directory, 'etrajectory.dcd')  # equilibration (protein-only)
        self.Eenergy      = join(directory, 'eenergy.csv')      # NPT equilibration energy trace
        self.Info         = join(directory, 'production.log')   # production state log
        self.Einfo        = join(directory, 'equilibration.log')  # equilibration state log

        # prepare()
        self.PdbFile = pdbFile
        self.Write = write

        # setup()
        self.MainForcefield = mainForcefield
        self.WaterForcefield = waterForcefield
        self.Stepsize = stepsize
        self.Integrator = openmm.VerletIntegrator(self.Stepsize)

        # minimize()
        self.Padding = padding
        self.BoxShape = boxShape

        # equilibrate()
        self.Temperature = temperature
        self.Pressure = pressure
        self.NonbondedMethod = nonbondedMethod
        self.NonbondedCutoff = nonbondedCutoff
        self.Constraints = constraints
        self.Friction = friction
        
        ## NVT
        self.EtimeNVT = etimeNVT
        self.EstepNVT = estepNVT
        
        ## NPT
        self.EtimeNPT = etimeNPT
        self.EstepNPT = estepNPT
        
        ## NPT loop
        self.PlateauTolerance = plateauTolerance
        self.SampleInterval = sampleInterval
        self.SmoothFraction = smoothFraction
        self.MaxEquilChunks = maxEquilChunks

        # run()
        self.Steps = steps
        self.LogInterval = logInterval
        self.Interval = interval

        # populated as the pipeline runs
        self.Fixer = None
        self.File = None
        self.Modeller = None
        self.Forcefield = None
        self.System = None
        self.Simulation = None
        self.SubsetTopology = None   # protein-only topology, shared by both trajectories
        self.SubsetIndices = None

        # lazily built on first access to self.analyze (run.analyze.<method>); the
        # import is deferred so a sim-only job never loads MDAnalysis.
        self._analyze = None

    @property
    def analyze(self):
        '''`analyze` handle bound to this run's model.pdb + trajectory.dcd, so a finished
        run can be analyzed in place with the same verb as the class:
        `sim.analyze.resCor("A", "B")`. Equivalent to
        `cam.analyze(sim.ModelPdb, sim.Trajectory)`. The import is deferred (handle built
        on first access), so a sim-only process never loads MDAnalysis -- touch it only once
        the run has frames.'''
        if self._analyze is None:
            from ._analyze import analyze
            self._analyze = analyze(self.ModelPdb, self.Trajectory,
                                    outputName=self.Directory)
        return self._analyze

    def prepare(self):
        '''Load and fix self.PdbFile (missing residues/atoms/hydrogens at pH 7).
        Stores self.Fixer always; writes the fixed structure to disk only if self.Write=True.'''
        base, ext = os.path.splitext(self.PdbFile)
        ext = ext.lower()
        if ext not in ('.pdb', '.cif'):
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info(f"Fixing {self.PdbFile}..")
        fixer = PDBFixer(filename=self.PdbFile)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        self.Fixer = fixer
        self.File = os.path.join(self.Directory, os.path.basename(base) + f'_fixed{ext}')

        if self.Write:
            writer = PDBxFile if ext == '.cif' else PDBFile
            with open(self.File, 'w') as fh:
                writer.writeFile(fixer.topology, fixer.positions, fh)
            logger.info(f"Wrote {self.File}")

        logger.info(f"Fixed {self.PdbFile}. Missing residues: {fixer.missingResidues}")

    def setup(self):
        '''Build modeller, forcefield, and minimization-ready system from self.Fixer'''
        self.Modeller = Modeller(self.Fixer.topology, self.Fixer.positions)
        self.Forcefield = ForceField(self.MainForcefield, self.WaterForcefield)
        self.System = self.Forcefield.createSystem(self.Modeller.topology)

    def minimize(self):
        '''Minimize the solute, then add a water box around the minimized positions'''
        logger.info("Minimizing solute..")
        minimizer = Simulation(self.Modeller.topology, self.System, self.Integrator,
                               platform=_cudaPlatform())
        minimizer.context.setPositions(self.Modeller.positions)
        minimizer.minimizeEnergy()
        self.Modeller.positions = minimizer.context.getState(getPositions=True).getPositions()
        logger.info("Solute minimized")

        logger.info("Adding water box..")
        self.Modeller.addSolvent(self.Forcefield, padding=self.Padding, boxShape=self.BoxShape)
        logger.info("Water box added")

    def equilibrate(self):
        '''Build the production system (nonbondedMethod, constraints, barostat), minimize the
        solvated system, then equilibrate in two phases: NVT warm-up to
        self.Temperature, followed by NPT with the barostat enabled.'''
        self.System = self.Forcefield.createSystem(self.Modeller.topology,
                                                     nonbondedMethod=self.NonbondedMethod,
                                                     nonbondedCutoff=self.NonbondedCutoff,
                                                     constraints=self.Constraints)

        self.Barostat = openmm.MonteCarloBarostat(self.Pressure, self.Temperature, 0)
        self.System.addForce(self.Barostat)

        self.Integrator = openmm.LangevinMiddleIntegrator(self.Temperature, self.Friction, self.Stepsize)
        self.Simulation = Simulation(self.Modeller.topology, self.System, self.Integrator,
                                     platform=_cudaPlatform())
        self.Simulation.context.setPositions(self.Modeller.positions)

        logger.info("Minimizing solvated system..")
        self.Simulation.minimizeEnergy()
        state = self.Simulation.context.getState(getEnergy=True, getPositions=True)
        logger.info(f"Energy: {state.getPotentialEnergy()}")

        # Protein-only topology (solvent stripped), shared by both stripped
        # trajectories. Written once as model.pdb -- the topology for analysis.
        self.SubsetTopology, self.SubsetIndices = self._proteinSubset()
        subsetPositions = state.getPositions(asNumpy=True)[self.SubsetIndices, :]
        with open(self.ModelPdb, 'w') as fh:
            PDBFile.writeFile(self.SubsetTopology, subsetPositions, fh)

        # Persist the restart artifacts now that the production system exists:
        # the serialized System (forces + barostat), the FULL solvated topology
        # (restart needs every atom, unlike the stripped model.pdb), and the run
        # manifest. With the rolling state.xml written during production, these
        # let restart() rebuild the Simulation in a fresh process on any GPU/host.
        with open(self.SystemXml, 'w') as fh:
            fh.write(openmm.XmlSerializer.serialize(self.System))
        with open(self.SolvatedPdb, 'w') as fh:
            PDBFile.writeFile(self.Modeller.topology, state.getPositions(asNumpy=True), fh)
        self._writeParams()

        if self.EtimeNVT == 0 and self.EtimeNPT == 0:
            return

        self.Simulation.reporters.append(
            StateDataReporter(self.Einfo, reportInterval=50, step=True, temperature=True,
                              volume=True, potentialEnergy=True, speed=True))
        self.Simulation.reporters.append(
            SubsetDCDReporter(self.Etrajectory, 500, self.SubsetTopology, self.SubsetIndices))

        stepsizeFs = self.Stepsize.value_in_unit(femtoseconds)

        if self.EtimeNVT > 0:
            logger.info("First equilibration (NVT warm-up)..")
            nIncrements = round(self.Temperature.value_in_unit(kelvin) / self.EstepNVT.value_in_unit(kelvin))
            totalStepsNVT = round(self.EtimeNVT * 1000 / stepsizeFs)
            tsteps = totalStepsNVT // nIncrements
            T = self.EstepNVT
            for i in range(nIncrements):
                self.Integrator.setTemperature(T)
                self.Simulation.step(tsteps)
                T += self.EstepNVT

        if self.EtimeNPT > 0:
            # NPT runs in chunks until the smoothed potential energy plateaus
            # (relative drift < plateauTolerance). The full trace is written to
            # self.Eenergy. RECOMMENDED: inspect that equilibration curve before
            # launching production -- automated plateau detection is a heuristic,
            # not a guarantee. Stiff or slowly-relaxing systems may need a larger
            # plateauTolerance window (smaller value) or a longer etimeNPT per
            # chunk to avoid declaring convergence while still drifting.
            logger.info("Second equilibration (NPT, barostat enabled)..")
            self.Barostat.setFrequency(self.EstepNPT)

            chunkSteps = round(self.EtimeNPT * 1000 / stepsizeFs)
            nSamples = max(chunkSteps // self.SampleInterval, 2)
            sampleTimePs = self.SampleInterval * stepsizeFs / 1000  # ps between energy samples
            window = max(1, min(round(self.SmoothFraction * nSamples), nSamples))
            kernel = np.ones(window) / window
            offset = (window - 1) // 2  # centering offset of the 'valid' moving average

            with open(self.Eenergy, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time_ps', 'potential_energy_kJ_mol', 'smoothed_kJ_mol', 'chunk'])

                elapsedPs = 0.0
                converged = False
                for chunk in range(self.MaxEquilChunks):
                    # Sample the potential energy across one chunk of NPT.
                    times, energies = [], []
                    for _ in range(nSamples):
                        self.Simulation.step(self.SampleInterval)
                        elapsedPs += sampleTimePs
                        e = self.Simulation.context.getState(getEnergy=True).getPotentialEnergy()
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

                    if relativeDrift < self.PlateauTolerance:
                        logger.info("NPT equilibration converged (energy plateaued)")
                        converged = True
                        break

                if not converged:
                    logger.warning(f"NPT equilibration did not plateau within {self.MaxEquilChunks} chunks")

    def _proteinSubset(self):
        '''Topology + original-topology atom indices for the non-solvent
        (protein/ligand) atoms, used to strip water/ions from the trajectory.'''
        modeller = Modeller(self.Modeller.topology, self.Modeller.positions)
        solvent = [a for a in modeller.topology.atoms() if a.residue.name in SOLVENT_RESIDUES]
        modeller.delete(solvent)
        indices = [a.index for a in self.Modeller.topology.atoms()
                   if a.residue.name not in SOLVENT_RESIDUES]
        return modeller.topology, indices

    def _writeParams(self):
        '''Write the restart manifest / run record. Holds everything restart()
        needs that is NOT already captured by system.xml (the integrator config)
        or solvated.pdb (the topology): integrator parameters and run config.'''
        params = {
            'pdbFile': self.PdbFile,
            'stepsize_fs': self.Stepsize.value_in_unit(femtoseconds),
            'temperature_K': self.Temperature.value_in_unit(kelvin),
            'friction_per_ps': self.Friction.value_in_unit(picosecond**-1),
            'pressure_atm': self.Pressure.value_in_unit(atmospheres),
            'steps': self.Steps,
            'interval': self.Interval,
            'logInterval': self.LogInterval,
        }
        with open(self.ParamsJson, 'w') as fh:
            json.dump(params, fh, indent=2)

    def _resume(self, targetSteps, interval, logInterval, append):
        '''Attach production reporters and step until currentStep == targetSteps.

        Reporter order is deliberate: the trajectory writer runs BEFORE the
        state writer, so a crash between them re-runs (at most) the last frame on
        restart rather than dropping it -- a duplicate frame is preferred over a
        gap. Both fire every `interval` steps so the trajectory and state.xml
        stay in lockstep (their restart points can never diverge).'''
        self.Simulation.reporters = []
        self.Simulation.reporters.append(
            StateDataReporter(self.Info, reportInterval=logInterval, step=True, temperature=True,
                              volume=True, potentialEnergy=True, speed=True, append=append))
        self.Simulation.reporters.append(
            SubsetDCDReporter(self.Trajectory, interval, self.SubsetTopology,
                              self.SubsetIndices, append=append))
        self.Simulation.reporters.append(StateXMLReporter(self.StateXml, interval))

        remaining = targetSteps - self.Simulation.currentStep
        if remaining <= 0:
            logger.info("Target step count already reached; nothing to run")
            return
        logger.info(f"Running production ({remaining} steps remaining)..")
        self.Simulation.step(remaining)

    def run(self):
        '''Run the production simulation for self.Steps timesteps.

        The simulation itself runs in explicit solvent (water + ions); only the
        logging is solvent-stripped. The solvent is omitted from model.pdb and
        trajectory.dcd to save space -- every downstream analysis is protein-only.
        model.pdb (the trajectory's topology) is written in equilibrate().'''
        self.Simulation.currentStep = 0
        # Reset the clock so state.xml's elapsed time measures PRODUCTION progress
        # (restart() derives the resume step from it; equilibration time must not
        # leak in).
        self.Simulation.context.setTime(0 * picosecond)
        # Write an initial restart point so even a crash within the first
        # `interval` can resume from production start.
        self.Simulation.saveState(self.StateXml)

        self._resume(self.Steps, self.Interval, self.LogInterval, append=False)

    def _load(self):
        '''Rebuild the Simulation in a fresh process from the persisted run dir
        and load the last saved state.xml. Shared by restart() and extend().
        Returns the parsed params.json manifest.

        Resumes PRODUCTION ONLY: if state.xml is absent, equilibration never
        finished -- resubmit the initial job for this design (equilibration is
        cheap relative to production).'''
        if not os.path.exists(self.StateXml):
            raise FileNotFoundError(
                f"No {self.StateXml}: production never started (equilibration "
                f"did not finish). Resubmit the initial job for this design.")

        with open(self.ParamsJson) as fh:
            p = json.load(fh)

        pdb = PDBFile(self.SolvatedPdb)
        with open(self.SystemXml) as fh:
            self.System = openmm.XmlSerializer.deserialize(fh.read())

        integrator = openmm.LangevinMiddleIntegrator(
            p['temperature_K'] * kelvin,
            p['friction_per_ps'] / picosecond,
            p['stepsize_fs'] * femtoseconds)
        self.Simulation = Simulation(pdb.topology, self.System, integrator,
                                     platform=_cudaPlatform())
        self.Simulation.loadState(self.StateXml)

        # loadState restores the context clock but NOT simulation.currentStep, so
        # derive production progress from the elapsed time (reset to 0 in run()).
        elapsedPs = self.Simulation.context.getState().getTime().value_in_unit(picosecond)
        self.Simulation.currentStep = round(elapsedPs * 1000 / p['stepsize_fs'])

        # Re-derive the protein-only subset for the appending trajectory writer.
        self.Modeller = Modeller(pdb.topology, pdb.positions)
        self.SubsetTopology, self.SubsetIndices = self._proteinSubset()
        return p

    def restart(self):
        '''Resume production in self.Directory toward its existing step target,
        from the last saved state.xml (saveState format -- portable GPU/host).'''
        p = self._load()
        logger.info(f"Restarting at step {self.Simulation.currentStep}; target {p['steps']}")
        self._resume(p['steps'], p['interval'], p['logInterval'], append=True)

    def extend(self):
        '''Continue a run toward a NEW, larger total step target (self.Steps),
        e.g. 100 ns -> 150 ns. The new target is persisted to params.json so
        subsequent restarts resume toward it. Same resume machinery as restart;
        only the target (and the manifest) change.'''
        p = self._load()
        if self.Steps <= p['steps']:
            raise ValueError(
                f"--extend target {self.Steps} must exceed the current target "
                f"{p['steps']}; pass a larger --steps")

        p['steps'] = self.Steps
        with open(self.ParamsJson, 'w') as fh:
            json.dump(p, fh, indent=2)

        logger.info(f"Extending from step {self.Simulation.currentStep} to {self.Steps}")
        self._resume(self.Steps, p['interval'], p['logInterval'], append=True)
