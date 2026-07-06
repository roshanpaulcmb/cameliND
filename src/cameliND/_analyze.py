'''_analyze.py

Trajectory analysis for cameliND simulations, built on MDAnalysis.

`analyze` is file-based: give it any topology (pdb/cif) + trajectory (dcd) and it
exposes the full analysis suite. A cameliND `simulate` run carries a same-named `analyze`
property bound to its own conventional paths, so a finished run is analyzed in place with
the same verb as the class:

    import cameliND as cam
    sim = cam.simulate(directory)
    sim.run()
    sim.analyze.findResCor("A", "B")     # == cam.analyze(sim.model_pdb, sim.trajectory)

Someone analyzing their own simulation uses the same class directly:

    a = cam.analyze("their.pdb", "their.dcd")
    a.runPca()
    a.findResCor("A", "B")            # receptor=A, nanobody=B

Each chain's heavy-atom group is keyed by id in self.chain ("A", "B", ...); the
binding analyses (findContacts/findDist/findContacts2/findResCor) take the two
chain ids to compare, so any pair from a many-chain topology works.

Every method computes and STORES its result on self (e.g. self.pc, self.resCor,
self.bindingScore); plotting is intentionally separate (see the plot() stub).
'''

#### IMPORTS ####
import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # backend left to matplotlib: inline in Jupyter, Agg when headless
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts, distances
from MDAnalysis.analysis.align import AlignTraj   # imported directly: `align` is a method name below
from MDAnalysis.exceptions import NoDataError
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import umap
import hdbscan

#### GLOBAL ####

logger = logging.getLogger(__name__)


#### CLASS ####

class analyze:
    '''Analyze a single MD trajectory (any topology + dcd).

    The binding analyses (findContacts/findDist/findContacts2/findResCor) treat
    chainA as the receptor (RBD) and chainB as the nanobody (NB).
    '''

    def __init__(self, topology, trajectory, outputName="", verbose=False):
        self.topology = topology        # any pdb/cif (model.pdb for a cameliND run)
        self.trajectory = trajectory    # any dcd
        self.outputName = outputName    # filename prefix reserved for plot()
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.INFO)

        logger.info("Making universe..")
        self.u = mda.Universe(topology, trajectory)
        self._refreshSelections()

        # populated by the analysis methods
        self.rmsdFirst = None
        self.rmsdLast = None
        self.rmsf = None
        self.rmsfProtein = None
        self.pc = None
        self.pcaModel = None
        self.sfePca = None
        self.sfeUmap = None
        self.umap = None
        self.cluster = None
        self.contacts = None
        self.distance = None
        self.filteredDistance = None
        self.contactFreq = None
        self.resCor = None
        self.attractionScore = None
        self.repulsionScore = None
        self.bindingScore = None

    def _detectChains(self):
        '''(selection keyword, sorted unique ids) for the protein's chains.
        Prefers the chainID attribute, falling back to segid.'''
        try:
            ids = self.protein.chainIDs
            key = "chainID"
        except (AttributeError, NoDataError):
            ids = self.protein.segids
            key = "segid"
        return key, sorted({c for c in ids if str(c).strip()})

    def _group(self, chainId):
        '''Heavy-atom AtomGroup for a chain id, with a clear error if absent.'''
        if chainId not in self.chain:
            raise ValueError(f"Chain {chainId!r} not found; available chains: {list(self.chain)}")
        return self.chain[chainId]

    def _refreshSelections(self):
        '''(Re)build self.protein and the per-chain heavy-atom groups from self.u.

        Per-chain heavy-atom groups are keyed by chain id (self.chain["A"], etc.);
        the binding analyses take two chain ids and look their groups up here.
        Called by __init__ and after downsize()/align() swap the trajectory.'''
        self.protein = self.u.select_atoms("protein")
        self.chainKey, chains = self._detectChains()
        self.chain = {c: self.u.select_atoms(f"protein and {self.chainKey} {c} and not name H*")
                      for c in chains}
        logger.info("Chains: %s", list(self.chain))

    #### TRAJECTORY PREP ####

    def downsize(self, percentage=10, write=True):
        '''Keep every Nth frame (0, N, 2N, ...) with N = round(100/percentage),
        reducing the trajectory to ~percentage% of its frames. The whole system is
        kept (no atom subset), so the original topology still matches.

        write=True (default) writes {outputName}_downsampled{percentage}.dcd and
        reloads from it; write=False subsamples into memory. Either way self.u and
        the selections are rebuilt. Returns self.'''
        step = max(1, round(100 / percentage))
        nBefore = len(self.u.trajectory)
        logger.info("Downsizing to %d%% (every %d frames) from %d frames..",
                    percentage, step, nBefore)

        if write:
            out = f"{self.outputName}_downsampled{percentage}.dcd"
            with mda.Writer(out, self.u.atoms.n_atoms) as w:
                for _ in self.u.trajectory[::step]:
                    w.write(self.u.atoms)
            self.trajectory = out
            self.u = mda.Universe(self.topology, out)
        else:
            self.u.transfer_to_memory(step=step)

        self._refreshSelections()
        logger.info(" Downsized: %d -> %d frames", nBefore, len(self.u.trajectory))
        return self

    def align(self, refFrame=0, select="backbone", chain=None, write=True):
        '''Superpose the trajectory onto a reference frame to remove rigid-body
        drift. Defaults to all-protein backbone against frame 0; pass a chain id to
        align on a single chain's backbone (e.g. hold the receptor fixed).

        write=True (default) writes {outputName}_aligned.dcd and reloads from it;
        write=False aligns in memory. Either way self.u and the selections are
        rebuilt. Returns self.'''
        if chain is not None:
            self._group(chain)   # validates the chain id
            select = f"{select} and {self.chainKey} {chain}"
        logger.info("Aligning on '%s' to frame %d..", select, refFrame)

        if write:
            out = f"{self.outputName}_aligned.dcd"
            AlignTraj(self.u, self.u, select=select, ref_frame=refFrame,
                      filename=out).run(verbose=self.verbose)
            self.trajectory = out
            self.u = mda.Universe(self.topology, out)
        else:
            self.u.transfer_to_memory()
            AlignTraj(self.u, self.u, select=select, ref_frame=refFrame,
                      in_memory=True).run(verbose=self.verbose)

        self._refreshSelections()
        return self

    #### WHOLE-PROTEIN ANALYSES ####

    def calcRmsd(self, select="name CA"):
        '''RMSD of `select` vs the first and the last frame. Stores
        self.rmsdFirst/rmsdLast (each a [3, n_frames] array: [frame, time, rmsd]).'''
        logger.info("Calculating RMSD..")
        rFirst = rms.RMSD(self.protein, self.protein,
                          select=select, ref_frame=0).run(verbose=self.verbose)
        self.rmsdFirst = rFirst.results.rmsd.T

        rLast = rms.RMSD(self.protein, self.protein,
                         select=select, ref_frame=-1).run(verbose=self.verbose)
        self.rmsdLast = rLast.results.rmsd.T
        return self

    def calcRmsf(self, select="name CA"):
        '''Per-residue `select` RMSF (self.rmsf) and per-atom protein RMSF
        (self.rmsfProtein). self.select holds the `select` AtomGroup for residue ids.'''
        logger.info("Calculating RMSF..")
        self.select = self.protein.select_atoms(select)
        self.rmsf = rms.RMSF(self.select).run(verbose=self.verbose).results.rmsf

        self.rmsfProtein = rms.RMSF(self.protein).run(verbose=self.verbose).results.rmsf
        self.protein.atoms.tempfactors = self.rmsfProtein
        return self

    def runPca(self, select="name CA"):
        '''PCA on flattened `select` coordinates (scikit-learn). Stores self.pc
        (principal components per frame) and self.pcaModel (the fitted PCA).'''
        logger.info("Running PCA..")
        ca = self.u.select_atoms(select)
        n_frames = len(self.u.trajectory)
        n_atoms = len(ca)

        coords = np.zeros((n_frames, n_atoms * 3), dtype=np.float32)
        for i, ts in enumerate(self.u.trajectory):
            coords[i, :] = ca.positions.reshape(-1)

        coords_centered = coords - coords.mean(axis=0)

        pca = PCA()
        self.pc = pca.fit_transform(coords_centered)
        self.pcaModel = pca

        logger.info(" Finished PCA: %d frames, %d components",
                    self.pc.shape[0], self.pc.shape[1])
        return self

    @staticmethod
    def _freeEnergy(states, T, kB, sigma):
        '''ΔF (kcal/mol) from a 2D histogram of state counts. The density is smoothed
        first, then converted with -kT ln(p): this spreads counts into neighbouring
        bins (forming real basins) and avoids flattening. Sign convention: the
        most-populated state sits at ΔF = 0 and everything else is negative, so
        high-probability basins read high (yellow) against a low (purple) background.'''
        density = gaussian_filter(states, sigma=sigma)
        prob = density / density.sum()
        with np.errstate(divide="ignore"):
            F = -kB * T * np.log(prob)            # J/molecule; empty bins -> +inf
        F *= 6.02214076E23 / 4184.0               # J/molecule -> kcal/mol
        finite = np.isfinite(F)
        F[~finite] = F[finite].max()              # bound unvisited bins at highest sampled energy
        return F.min() - F                        # ΔF in [negative, 0]: wells at 0, rest negative

    def calcSfe(self, components=(0, 1), bins=200, sigma=2, T=310, kB=1.380649E-23):
        '''Free-energy surface (kcal/mol) over two principal components, smoothed.
        Stores self.sfePca. T in Kelvin; kB in J/K.'''
        logger.info("Calculating surface free energy..")

        states, xedges, yedges = np.histogram2d(self.pc[:, components[0]],
                                                self.pc[:, components[1]], bins=bins)
        self.sfePca = self._freeEnergy(states, T, kB, sigma)
        return self

    def runUmap(self, n_neighbors=90, min_dist=0.6, n_components=3,
                n_pcs=10, metric="euclidean", random_state=42):
        '''UMAP embedding of the first n_pcs PCs. Stores self.umap.'''
        logger.info("Running UMAP..")
        start = time.perf_counter()

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        self.umap = reducer.fit_transform(self.pc[:, :n_pcs])

        logger.info("Finished UMAP in %.2f seconds", time.perf_counter() - start)
        return self

    def runUmapSfe(self, components=(0, 1), bins=200, sigma=2, T=310, kB=1.380649E-23):
        '''Free-energy surface (kcal/mol) over two UMAP dimensions, smoothed.
        Stores self.sfeUmap. T in Kelvin; kB in J/K.'''
        logger.info("Running UMAP SFE..")

        states, xedges, yedges = np.histogram2d(self.umap[:, components[0]],
                                                self.umap[:, components[1]], bins=bins)
        self.sfeUmap = self._freeEnergy(states, T, kB, sigma)
        return self

    def clusterUmap(self, min_cluster_size=500, min_samples=1,
                    cluster_selection_epsilon=0.01):
        '''HDBSCAN clustering of the UMAP embedding; noise points (-1) are
        reassigned to the nearest cluster centroid. Stores self.cluster.'''
        logger.info("Clustering UMAP..")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        labels = clusterer.fit_predict(self.umap)

        # Assign noise points (-1) to nearest cluster
        noise_idx = np.where(labels == -1)[0]
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) == 0:
            logger.warning("No clusters found (min_cluster_size=%d too large for %d points); "
                           "all points left as noise", min_cluster_size, len(labels))
        elif len(noise_idx) > 0:
            centroids = np.array([self.umap[labels == lbl].mean(axis=0) for lbl in unique_labels])
            for i in noise_idx:
                nearest = np.argmin(cdist([self.umap[i]], centroids))
                labels[i] = unique_labels[nearest]

        self.cluster = labels
        return self

    #### BINDING-PAIR ANALYSES ####

    def findContacts(self, chain0, chain1, cutoff=4.5, method="radius_cut"):
        '''Fraction of native contacts between two chains over the trajectory.
        Stores self.contacts (DataFrame: Frame, Contacts).'''
        logger.info("Finding contact frequency over trajectory..")
        groupA, groupB = self._group(chain0), self._group(chain1)
        selA = f"protein and {self.chainKey} {chain0} and not name H*"
        selB = f"protein and {self.chainKey} {chain1} and not name H*"
        c = contacts.Contacts(self.u,
                              select=(selA, selB),
                              refgroup=(groupA, groupB),
                              radius=cutoff,
                              method=method)
        c.run(verbose=self.verbose)
        self.contacts = pd.DataFrame(c.results.timeseries, columns=["Frame", "Contacts"])
        return self

    def findDist(self, chain0, chain1, cutoff=4.5):
        '''Frame-averaged residue-residue distance matrix (chain1 x chain0) plus a
        cutoff-masked copy. Stores self.distance and self.filteredDistance.'''
        logger.info("Finding average pairwise distances..")
        comA = self._group(chain0).center_of_mass(compound="residues")
        comB = self._group(chain1).center_of_mass(compound="residues")

        dL = []
        start = time.perf_counter()
        for step in self.u.trajectory:
            d = distances.distance_array(comB, comA, box=self.u.dimensions)
            dL.append(d)
        logger.info("Finished distances in %.2f seconds", time.perf_counter() - start)

        self.distance = np.mean(np.array(dL), axis=0)
        mask = self.distance <= cutoff
        self.filteredDistance = np.where(mask, self.distance, 0)
        return self

    def findContacts2(self, chain0, chain1, cutoff=4.5):
        '''Per-residue-pair contact frequency (chain1 x chain0): fraction of frames
        the residue COMs are within cutoff. Stores self.contactFreq.'''
        logger.info("Finding pairwise contacts..")
        comA = self._group(chain0).center_of_mass(compound="residues")
        comB = self._group(chain1).center_of_mass(compound="residues")

        contactA = np.zeros((len(comB), len(comA)))
        start = time.perf_counter()
        for step in self.u.trajectory:
            d = distances.distance_array(comB, comA, box=self.u.dimensions)
            contactA += (d < cutoff)
        logger.info("Finished contacts in %.2f seconds", time.perf_counter() - start)

        self.contactFreq = contactA / len(self.u.trajectory)
        return self

    # https://pubs.acs.org/doi/10.1021/acs.jcim.5c01725#fig1
    def findResCor(self, chain0, chain1, attractiveCutoff=8, repulsiveCutoff=13):
        '''Normalized inter-chain CA displacement correlation, the binding score.

        Builds the close-contact correlation matrix (self.resCor): positive
        (attractive) correlations kept within attractiveCutoff A, negative
        (repulsive) within repulsiveCutoff A. Stores self.attractionScore,
        self.repulsionScore, self.bindingScore.

        Uses RAW positions (the trajectory must be unwrapped); per-molecule wrapping
        would teleport a chain across the box and corrupt the correlation.'''
        logger.info("Finding inter-chain residue correlation..")
        caA = self._group(chain0).select_atoms("name CA")
        caB = self._group(chain1).select_atoms("name CA")

        nB = len(caB)
        nA = len(caA)
        nFrames = len(self.u.trajectory)

        # Average position of each residue's CA over the trajectory
        bAvgPos = np.zeros((nB, 3))
        aAvgPos = np.zeros((nA, 3))
        for step in self.u.trajectory:
            bAvgPos += caB.positions
            aAvgPos += caA.positions
        bAvgPos /= nFrames
        aAvgPos /= nFrames

        # Inter-chain covariance C(i,j) = < dR_i . dR_j >
        cov = np.zeros((nB, nA))
        varB = np.zeros(nB)
        varA = np.zeros(nA)
        for step in self.u.trajectory:
            bDelta = caB.positions - bAvgPos
            aDelta = caA.positions - aAvgPos
            cov += bDelta @ aDelta.T
            varB += np.sum(bDelta ** 2, axis=1)
            varA += np.sum(aDelta ** 2, axis=1)
        cov /= nFrames
        varB /= nFrames
        varA /= nFrames

        # Normalize: C(i,j) / sqrt(var_i * var_j)
        norm = np.zeros_like(cov)
        for i in range(nB):
            for j in range(nA):
                denom = np.sqrt(varB[i] * varA[j])
                norm[i, j] = cov[i, j] / denom if denom != 0 else 0

        # Average residue-residue distances for the close-contact cutoff
        avgD = np.zeros((nB, nA))
        for i in range(nB):
            for j in range(nA):
                avgD[i, j] = np.linalg.norm(bAvgPos[i] - aAvgPos[j])

        # attractiveCutoff for positive correlations, repulsiveCutoff for negative
        cutoffM = np.zeros_like(norm)
        cutoffM[(norm > 0) & (avgD <= attractiveCutoff)] = norm[(norm > 0) & (avgD <= attractiveCutoff)]
        cutoffM[(norm < 0) & (avgD <= repulsiveCutoff)] = norm[(norm < 0) & (avgD <= repulsiveCutoff)]

        self.resCor = cutoffM
        self.attractionScore = np.sum(cutoffM[cutoffM > 0])
        self.repulsionScore = np.sum(cutoffM[cutoffM < 0])
        self.bindingScore = np.sum(cutoffM)
        return self

    #### ORCHESTRATION ####

    def runAll(self, chain0=None, chain1=None):
        '''Run the full suite. The whole-protein analyses always run; the binding
        analyses run on (chain0, chain1) -- defaulting to the two chains when the
        topology has exactly two, and skipped (with a warning) otherwise.'''
        self.calcRmsd()
        self.calcRmsf()
        self.runPca()
        self.calcSfe()
        self.runUmap()
        self.runUmapSfe()
        self.clusterUmap()

        if chain0 is None and chain1 is None and len(self.chain) == 2:
            chain0, chain1 = list(self.chain)
        if chain0 is not None and chain1 is not None:
            self.findContacts(chain0, chain1)
            self.findDist(chain0, chain1)
            self.findContacts2(chain0, chain1)
            self.findResCor(chain0, chain1)
        else:
            logger.warning("Skipping binding analyses: pass chain0, chain1 (chains: %s)",
                           list(self.chain))
        return self

    #### PLOTTING ####

    def _require(self, value, method):
        '''Guard: a plotter's result must exist or we name the method to run.'''
        if value is None:
            raise ValueError(f"Nothing to plot; run {method} first")

    def _heatmap(self, mat, title, xlabel, ylabel, cmap, cbar, **kw):
        '''Shared 2D-matrix figure (free-energy surfaces, distance/contact maps).'''
        fig, ax = plt.subplots()
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, **kw)
        fig.colorbar(im, ax=ax, label=cbar)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        return fig, ax

    def _plotRmsd(self):
        self._require(self.rmsdFirst, "calcRmsd()")
        fig, ax = plt.subplots()
        ax.plot(self.rmsdFirst[1], self.rmsdFirst[2], label="vs first frame")
        ax.plot(self.rmsdLast[1], self.rmsdLast[2], label="vs last frame")
        ax.set(xlabel="Time (ps)", ylabel="RMSD (Å)", title="Backbone RMSD")
        ax.legend()
        return fig, ax

    def _plotRmsf(self):
        self._require(self.rmsf, "calcRmsf()")
        fig, ax = plt.subplots()
        # Plot against a continuous CA index, not resids: residue numbering resets
        # per chain, so resids are non-monotonic and would draw a line across the break.
        ax.plot(np.arange(len(self.rmsf)), self.rmsf)
        ax.set(xlabel="Residue (CA index)", ylabel="RMSF (Å)", title="Per-residue RMSF")
        return fig, ax

    def _plotSfePca(self):
        self._require(self.sfePca, "calcSfe()")
        return self._heatmap(self.sfePca.T, "PCA free-energy surface",
                             "PC1", "PC2", "viridis", "ΔF (kcal/mol)")

    def _plotSfeUmap(self):
        self._require(self.sfeUmap, "runUmapSfe()")
        return self._heatmap(self.sfeUmap.T, "UMAP free-energy surface",
                             "UMAP1", "UMAP2", "viridis", "ΔF (kcal/mol)")

    def _plotUmap(self):
        self._require(self.umap, "runUmap()")
        fig, ax = plt.subplots()
        sc = ax.scatter(self.umap[:, 0], self.umap[:, 1],
                        c=self.cluster, cmap="tab10", s=5)
        if self.cluster is not None:
            fig.colorbar(sc, ax=ax, label="cluster")
        ax.set(xlabel="UMAP1", ylabel="UMAP2", title="UMAP embedding")
        return fig, ax

    def _plotContacts(self):
        self._require(self.contacts, "findContacts()")
        fig, ax = plt.subplots()
        ax.plot(self.contacts["Frame"], self.contacts["Contacts"])
        ax.set(xlabel="Frame", ylabel="Fraction native contacts",
               title="Native contacts")
        return fig, ax

    def _plotDistance(self):
        self._require(self.distance, "findDist()")
        return self._heatmap(self.distance, "Mean residue–residue distance",
                             "chain0 residue", "chain1 residue", "viridis_r", "Distance (Å)")

    def _plotContactFreq(self):
        self._require(self.contactFreq, "findContacts2()")
        return self._heatmap(self.contactFreq, "Contact frequency",
                             "chain0 residue", "chain1 residue", "magma", "Contact frequency")

    def _plotResCor(self):
        self._require(self.resCor, "findResCor()")
        vmax = np.abs(self.resCor).max() or 1
        return self._heatmap(self.resCor, "Inter-chain CA correlation",
                             "chain0 residue", "chain1 residue", "bwr", "Correlation",
                             vmin=-vmax, vmax=vmax)

    def plot(self, kind, write=False, show=True):
        '''Render a stored result: display it (show=True) and/or write
        {outputName}{kind}.png (write=True); returns (fig, ax). Plotting is
        deliberately separate from the analysis methods (results live on self), so
        the matching method must have run first -- otherwise the error names it.
        Valid kinds are the keys below.'''
        plotters = {
            "rmsd": self._plotRmsd,
            "rmsf": self._plotRmsf,
            "sfe": self._plotSfePca,
            "sfeUmap": self._plotSfeUmap,
            "umap": self._plotUmap,
            "contacts": self._plotContacts,
            "distance": self._plotDistance,
            "contactFreq": self._plotContactFreq,
            "resCor": self._plotResCor,
        }
        if kind not in plotters:
            raise ValueError(f"Unknown plot {kind!r}; choose from {list(plotters)}")
        fig, ax = plotters[kind]()
        if write:
            fname = f"{self.outputName}{kind}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            logger.info("Wrote %s", fname)
        if show:
            plt.show()
        return fig, ax

    def plotAll(self, write=False, show=True):
        '''Plot every result computed so far, skipping the ones still None.'''
        available = {"rmsd": self.rmsdFirst, "rmsf": self.rmsf, "sfe": self.sfePca,
                     "sfeUmap": self.sfeUmap, "umap": self.umap, "contacts": self.contacts,
                     "distance": self.distance, "contactFreq": self.contactFreq,
                     "resCor": self.resCor}
        for kind, result in available.items():
            if result is not None:
                self.plot(kind, write=write, show=show)
        return self


#### CLI ####

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cameliND trajectory analysis")
    parser.add_argument("--pdb", required=True, help="Topology (pdb/cif), e.g. model.pdb")
    parser.add_argument("--dcd", required=True, help="Trajectory (dcd), e.g. trajectory.dcd")
    parser.add_argument("--outputName", default="", help="Filename prefix (reserved for plotting)")
    parser.add_argument("--chain0", help="First binding chain id (receptor); default: auto if 2 chains")
    parser.add_argument("--chain1", help="Second binding chain id (nanobody); default: auto if 2 chains")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    analyze(args.pdb, args.dcd, outputName=args.outputName, verbose=True).runAll(args.chain0, args.chain1)
