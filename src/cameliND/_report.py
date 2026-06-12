'''_report.py

Custom OpenMM reporters for the cameliND production pipeline (used by _simulate.py):
SubsetDCDReporter writes solvent-stripped trajectory frames; StateXMLReporter writes
crash-safe, portable restart points. Both follow OpenMM's
describeNextReport/report reporter protocol.
'''
import os

from openmm.app import DCDFile


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
