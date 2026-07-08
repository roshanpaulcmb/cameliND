# cameliND

Nanobody design *in silico* with AI and physics — inspired by nature's camelids.

## Workflow

The pipeline consists of three stages:

1. **Design** — Generate nanobody designs against a target (BoltzGen).
2. **Simulate** — Run molecular dynamics simulations on the designs (`cameliND.simulate`, OpenMM).
3. **Analyze** — Analyze simulation trajectories for stability and binding properties (`cameliND.analyze`, MDAnalysis).

Design and simulate/analyze run in **separate conda environments** (`boltzgen_env.yml`
and `cameliND_env.yml`); they sit at different pipeline stages and are never imported in
the same process.

## Installation

`cameliND` runs in a conda environment (it depends on OpenMM, MDAnalysis, and other
scientific packages). Create the environment, then install the package into it:

```bash
git clone <repo-url>
cd cameliND
conda env create -f cameliND_env.yml     # OpenMM, MDAnalysis, scikit-learn, ...
conda activate cameliND
pip install .                            # install the cameliND package
```

Design generation uses BoltzGen in its own separate environment (`boltzgen_env.yml`).

## The simulation pipeline

`cameliND.simulate` wraps an OpenMM molecular-dynamics workflow as a sequence of
stages. All run parameters are set once when the object is constructed; each stage then
advances the run from the input structure to a finished production trajectory:

```python
import cameliND as cam

s = cam.simulate(directory="run/dir", pdbFile="design.cif")
s.prepare()
s.setup()
s.minimize()
s.equilibrate()
s.run()
```

| Stage | What it does |
|---|---|
| `prepare()` | Repairs the input structure with PDBFixer — adds missing residues and atoms, and protonates at pH 7. |
| `setup()` | Builds the force field (Amber19 + TIP3P water) and the initial system. |
| `minimize()` | Energy-minimizes the solute, then surrounds it with a solvent (water + ions) box. |
| `equilibrate()` | Builds the production system (PME electrostatics, H-bond constraints, Monte Carlo barostat), then equilibrates: an NVT warm-up to temperature, followed by NPT held until the potential energy plateaus. Writes the protein-only topology (`model.pdb`) and the restart files. |
| `run()` | Runs production MD (default 100 ns at 2 fs steps). Writes a solvent-stripped `trajectory.dcd` plus rolling, portable checkpoints (`state.xml`) for crash-safe restarts. |

Each run lives in a self-describing directory (`params.json`, `system.xml`, `state.xml`,
`model.pdb`, `trajectory.dcd`, ...), so it can be resumed or extended from that directory
alone:

```python
cam.simulate(directory="run/dir").restart()                  # resume toward the existing target
cam.simulate(directory="run/dir", steps=75000000).extend() # continue toward a longer target
```

### `simulate()` inputs

All run parameters are passed to the constructor (units come from `openmm.unit`). Only
`directory` is required; `pdbFile` is required for a fresh run (not for `restart` /
`extend`). The force field, water model, box shape, and system options accept any value
OpenMM supports — see OpenMM's [force fields](http://docs.openmm.org/latest/userguide/application/02_running_sims.html#force-fields)
list and the [`Modeller.addSolvent`](http://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html#openmm.app.modeller.Modeller.addSolvent)
and [`ForceField.createSystem`](http://docs.openmm.org/latest/api-python/generated/openmm.app.forcefield.ForceField.html#openmm.app.forcefield.ForceField.createSystem)
API docs.

Each attribute is stored as the parameter input with a capital first letter (e.g.
`s.Directory`, `s.PdbFile`).

| Parameter | Default | Meaning |
|---|---|---|
| `directory` | *(required)* | Run directory; all artifacts are written here. |
| `pdbFile` | `None` | Input structure (`.pdb`/`.cif`) for a fresh run. |
| `write` | `False` | Also write the PDBFixer-repaired structure to disk. |
| `mainForcefield` | `'amber19-all.xml'` | Protein force field ([other options](http://docs.openmm.org/latest/userguide/application/02_running_sims.html#force-fields)). |
| `waterForcefield` | `'amber19/tip3pfb.xml'` | Water model ([other options](http://docs.openmm.org/latest/userguide/application/02_running_sims.html#force-fields)). |
| `stepsize` | `2*femtoseconds` | Integration timestep. |
| `padding` | `1.0*nanometer` | Solvent box padding around the solute. |
| `boxShape` | `'octahedron'` | Solvent box shape: `'cube'`, `'dodecahedron'`, or `'octahedron'` (see [`addSolvent`](http://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html#openmm.app.modeller.Modeller.addSolvent)). |
| `temperature` | `310*kelvin` | Production temperature. |
| `pressure` | `1*atmospheres` | Barostat pressure (NPT). |
| `nonbondedMethod` | `PME` | Long-range electrostatics method: `NoCutoff`, `CutoffNonPeriodic`, `CutoffPeriodic`, `Ewald`, `PME`, or `LJPME` (see [`createSystem`](http://docs.openmm.org/latest/api-python/generated/openmm.app.forcefield.ForceField.html#openmm.app.forcefield.ForceField.createSystem)). |
| `nonbondedCutoff` | `1*nanometer` | Nonbonded cutoff. |
| `constraints` | `HBonds` | Bond constraints: `None`, `HBonds`, `AllBonds`, or `HAngles` (`HBonds` enables the 2 fs step; see [`createSystem`](http://docs.openmm.org/latest/api-python/generated/openmm.app.forcefield.ForceField.html#openmm.app.forcefield.ForceField.createSystem)). |
| `friction` | `1/picosecond` | Langevin friction coefficient. |
| `etimeNVT` | `100` | NVT warm-up duration (ps); `0` skips NVT. |
| `etimeNPT` | `100` | NPT duration per equilibration chunk (ps); `0` skips NPT. |
| `estepNVT` | `3.1*kelvin` | Temperature increment per NVT warm-up step. |
| `estepNPT` | `25` | Barostat update frequency (steps) during NPT. |
| `plateauTolerance` | `5e-5` | NPT convergence: max fractional energy drift per ps. |
| `sampleInterval` | `500` | Steps between NPT energy samples. |
| `smoothFraction` | `0.1` | Moving-average window (fraction of samples per chunk). |
| `maxEquilChunks` | `10` | Cap on NPT equilibration chunks. |
| `steps` | `50000000` | Production steps (50M × 2 fs = 100 ns). |
| `logInterval` | `5000` | Steps between text state-log lines (~10 ps). |
| `interval` | `50000` | Steps between trajectory frames **and** checkpoints (~100 ps). |

## Analyzing a trajectory

There are two ways to get an `analyze` object:

1. **From a finished `simulate` run** — every `simulate` object exposes a bound analyzer
   as `s.analyze`, already wired to that run's `model.pdb` + `trajectory.dcd`. Analyze
   the simulation you just ran, in place:

   ```python
   s.analyze.resCor("A", "B")
   ```

2. **From files directly** — `analyze` is file-based, so you can point it at any topology
   (PDB/CIF) and trajectory (DCD) — e.g. a simulation you ran elsewhere:

   ```python
   a = cam.analyze("model.pdb", "trajectory.dcd")
   a.resCor("A", "B")
   ```

Either way you get the same `analyze` object with the same methods.

Methods are lowerCamelCase verbs; each computes and stores its result on the object as
the same name in UpperCamelCase (e.g. `resCor()` → `a.ResCor`, `pca()` → `a.Pc`, the
binding score in `a.BindingScore`). They fall into four groups:

- **Trajectory prep** (optional, run first) — `downsize()` keeps every Nth frame to
  thin a long trajectory; `align()` superposes every frame onto a reference to remove
  rigid-body drift. Both rebuild the universe in place and return `self`, so they chain
  (`a.downsize().align(chain="A")`); by default each writes the reduced/aligned DCD next to
  the input (`write=False` does it in memory instead).
- **Structural stability** — `rmsd()` (deviation from the reference over time) and
  `rmsf()` (per-residue flexibility).
- **Conformational landscape** — `pca()` (principal motions), `sfe()` (free-energy
  surface over the top components), `umap()` / `umapSfe()` (nonlinear embedding) and
  `clusterUmap()` (HDBSCAN clustering of conformational states).
- **Binding (receptor ↔ nanobody)** — `contacts()`, `dist()`, `contacts2()`
  (inter-chain contact and distance maps), and `resCor()`, the binding-score
  centerpiece: it measures normalized inter-chain Cα displacement correlation (attraction
  within 8 Å, repulsion within 13 Å). In a single pass it accumulates the score as a
  cumulative function of time — `a.BindingSeries` (with `a.AttractionSeries` /
  `a.RepulsionSeries`) against `a.BindingTime` — and the full-trajectory scalar
  `a.BindingScore` is just the last point of that curve. Pass `startFrame` (a frame
  index, or `"min"` to auto-detect the end of the docking-relaxation transient at the
  cumulative minimum) to compute the correlation over only the equilibrated window;
  `a.ResCorStartFrame` records the window used and `a.ResCorFlagged` marks designs
  whose interface never stabilized.

Binding analyses take the two chain IDs to compare, e.g. `a.resCor("A", "B")`
(receptor = A, nanobody = B).

### `analyze()` inputs

The constructor takes the topology and trajectory; chains are detected automatically:

| Parameter | Default | Meaning |
|---|---|---|
| `topology` | *(required)* | Topology file (`.pdb`/`.cif`). |
| `trajectory` | *(required)* | Trajectory file (`.dcd`). |
| `outputName` | `""` | Filename prefix for outputs. |
| `verbose` | `False` | Emit progress logging. |

Every tunable value in a method is a keyword argument:

| Method | Key arguments (defaults) |
|---|---|
| `downsize` | `percentage=10`, `write=True` |
| `align` | `refFrame=0`, `select="backbone"`, `chain=None`, `write=True` |
| `rmsd` / `rmsf` / `pca` | `select="name CA"` |
| `sfe` | `components=(0, 1)`, `bins=200`, `sigma=2`, `T=310`, `kB=1.380649e-23` |
| `umap` | `n_neighbors=90`, `min_dist=0.6`, `n_components=3`, `n_pcs=10`, `metric="euclidean"`, `random_state=42` |
| `umapSfe` | `components=(0, 1)`, `bins=200`, `sigma=2`, `T=310`, `kB=1.380649e-23` |
| `clusterUmap` | `min_cluster_size=500`, `min_samples=1`, `cluster_selection_epsilon=0.01` |
| `contacts` | `chain0`, `chain1`, `cutoff=4.5`, `method="radius_cut"` |
| `dist` / `contacts2` | `chain0`, `chain1`, `cutoff=4.5` |
| `resCor` | `chain0`, `chain1`, `attractiveCutoff=8`, `repulsiveCutoff=13`, `stride=1`, `minFrames=10`, `startFrame=0` |

`all(chain0=None, chain1=None)` runs the whole suite in dependency order; pass the two
chain IDs to include the binding analyses. (`T` in Kelvin, `kB` in J/K.)

### Plotting

Plotting is kept separate from the analyses — the methods only compute and store, so a
result must exist before it can be drawn (otherwise the error names the method to run).
`plot(kind, write=False, show=True)` renders one stored result and returns its
`(fig, ax)`; `plotAll(write=False, show=True)` draws every result computed so far,
skipping the ones not yet run. `write=True` saves `{outputName}{kind}.png`. Valid kinds:

| `kind` | Needs | Shows |
|---|---|---|
| `rmsd` | `rmsd()` | RMSD vs first/last frame over time |
| `rmsf` | `rmsf()` | per-residue flexibility |
| `sfe` / `sfeUmap` | `sfe()` / `umapSfe()` | free-energy surface (PCA / UMAP) |
| `umap` | `umap()` (+ `clusterUmap()`) | UMAP embedding, colored by cluster |
| `contacts` | `contacts()` | native-contact fraction over time |
| `distance` / `contactFreq` | `dist()` / `contacts2()` | inter-chain distance / contact-frequency map |
| `resCor` | `resCor()` | inter-chain Cα correlation matrix (condensed to interacting residues; `plot("resCor", view="all")` for the full matrix) |
| `bindingScore` | `resCor()` | attraction/repulsion/binding score vs time |

## Command line

The same pipeline is available as a CLI (used by the SLURM scripts):

```bash
python -m cameliND simulate --pdb design.cif --dir run/
python -m cameliND restart  --dir run/
python -m cameliND extend   --dir run/ --steps 75000000
```

## Development

Contributors working on the package should install it editable, so code changes take
effect without reinstalling. See [`docs/PACKAGING.md`](docs/PACKAGING.md) for how the
`src/` layout is put together.

```bash
pip install -e .
```

## Acknowledgements

This project builds on [BoltzGen](https://github.com/HannesStark/boltzgen) by Hannes Stark and colleagues. I gratefully acknowledge the BoltzGen team for their open-source release of model weights, data, and inference and training code. Please cite their work:

```bibtex
@article{stark2025boltzgen,
  author  = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and Xie, Yuxin and Hur, Eunsu and O'Donnell, Timothy John and Bushuiev, Anton and U\c{c}ar, Talip and Passaro, Saro and Mao, Weian and Reveiz, Mateo and Bushuiev, Roman and Pluskal, Tom\'a\v{s} and Sivic, Josef and Kreis, Karsten and Vahdat, Arash and Ray, Shamayeeta and Goldstein, Jonathan T. and Savinov, Andrew and Hambalek, Jacob A. and Gupta, Anshika and Taquiri-Diaz, Diego A. and Zhang, Yaotian and Hatstat, A. Katherine and Arada, Angelika and Kim, Nam Hyeong and Tackie-Yarboi, Ethel and Boselli, Dylan and Schnaider, Lee and Liu, Chang C. and Li, Gene-Wei and Hnisz, Denes and Sabatini, David M. and DeGrado, William F. and Wohlwend, Jeremy and Corso, Gabriele and Barzilay, Regina and Jaakkola, Tommi},
  title   = {BoltzGen: Toward Universal Binder Design},
  year    = {2025},
  doi     = {10.1101/2025.11.20.689494},
  journal = {bioRxiv}
}
```

Simulation uses **OpenMM** and trajectory analysis uses **MDAnalysis**. Please also cite:

```bibtex
@article{eastman2017openmm,
  author  = {Eastman, Peter and Swails, Jason and Chodera, John D. and McGibbon, Robert T. and Zhao, Yutong and Beauchamp, Kyle A. and Wang, Lee-Ping and Simmonett, Andrew C. and Harrigan, Matthew P. and Stern, Chaya D. and Wiewiora, Rafal P. and Brooks, Bernard R. and Pande, Vijay S.},
  title   = {{OpenMM} 7: Rapid development of high performance algorithms for molecular dynamics},
  journal = {PLOS Computational Biology},
  volume  = {13},
  number  = {7},
  pages   = {e1005659},
  year    = {2017},
  doi     = {10.1371/journal.pcbi.1005659}
}

@article{michaud2011mdanalysis,
  author  = {Michaud-Agrawal, Naveen and Denning, Elizabeth J. and Woolf, Thomas B. and Beckstein, Oliver},
  title   = {{MDAnalysis}: A toolkit for the analysis of molecular dynamics simulations},
  journal = {Journal of Computational Chemistry},
  volume  = {32},
  number  = {10},
  pages   = {2319--2327},
  year    = {2011},
  doi     = {10.1002/jcc.21787}
}

@inproceedings{gowers2016mdanalysis,
  author    = {Gowers, Richard J. and Linke, Max and Barnoud, Jonathan and Reddy, Tyler J. E. and Melo, Manuel N. and Seyler, Sean L. and Domanski, Jan and Dotson, David L. and Buchoux, S\'ebastien and Kenney, Ian M. and Beckstein, Oliver},
  title     = {{MDAnalysis}: A {Python} Package for the Rapid Analysis of Molecular Dynamics Simulations},
  booktitle = {Proceedings of the 15th Python in Science Conference},
  pages     = {98--105},
  year      = {2016},
  doi       = {10.25080/Majora-629e541a-00e}
}
```
