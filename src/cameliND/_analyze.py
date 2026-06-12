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
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts, distances
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
        self.protein = self.u.select_atoms("protein")

        # Per-chain heavy-atom groups, keyed by chain id: self.chain["A"], etc.
        # The binding analyses take two chain ids and look their groups up here.
        self.chainKey, chains = self._detectChains()
        self.chain = {c: self.u.select_atoms(f"protein and {self.chainKey} {c} and not name H*")
                      for c in chains}
        logger.info("Chains: %s", list(self.chain))

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

    def calcSfe(self, components=(0, 1), bins=200, sigma=2, T=310, kB=1.380649E-23):
        '''Free-energy surface over two principal components (smoothed).
        Stores self.sfePca. T in Kelvin (recommended); kB in J/K.'''
        logger.info("Calculating surface free energy..")

        states, xedges, yedges = np.histogram2d(self.pc[:, components[0]],
                                                self.pc[:, components[1]], bins=bins)
        prob = states / np.sum(states)
        F = -kB * T * np.log(prob, where=(prob > 0))
        deltaF = F - np.nanmin(F)
        self.sfePca = gaussian_filter(deltaF, sigma=sigma)
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
        '''Free-energy surface over two UMAP dimensions. Stores self.sfeUmap.
        T in Kelvin (recommended); kB in J/K.'''
        logger.info("Running UMAP SFE..")

        states, xedges, yedges = np.histogram2d(self.umap[:, components[0]],
                                                self.umap[:, components[1]], bins=bins)
        prob = states / np.sum(states)
        F = -kB * T * np.log(prob, where=(prob > 0))
        deltaF = F - np.nanmin(F)
        self.sfeUmap = gaussian_filter(deltaF, sigma=sigma)
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
        if len(noise_idx) > 0:
            unique_labels = np.unique(labels[labels != -1])
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

    def plot(self, kind):
        '''Render a stored result. Reserved single plotting entry point; plotting
        is deliberately separate from the analysis methods (results live on self).'''
        raise NotImplementedError("plot() is not implemented yet")


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
