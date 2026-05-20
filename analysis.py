#### IMPORTS ####

import argparse as arg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts, distances
import mdtraj as mdt
from scipy.ndimage import gaussian_filter
import os
import time
import umap
import hdbscan
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#### NOTES ####

# Urgent!!!
# 1. Label bands on residue correlation graph
# 2. Check figure size to 12, 9
# 3. Alter figure size for ace2
# 4. creaete functions that only show residues involved in distance, residue cor, etc. 

# To dos
# Push code to github
# Clean up functions. Start with changing naming convention
# 1. Perhaps changing data["rbd"] to the actual description "protein and chainID A and not name H*"
# 2. Clean up code. Ex, instances of where new variable is defined for data[key]. Just use data[key]
# 3. load in mdtraj in makeU. Perhas call that the setup function. No other file should require files

# Push to github again
# Test script on all the nb designed

# Long term
# Clean repository. Folders
# 1. Nb generation
# 2. code used to run simulations
# 3. analysis
# Write a graphing function, make if statement in functions if visualization == True: ...

#### FUNCTIONS ####


def setup():
    data = {}
    return data

def makeU(files, data):
    print("\nMaking universe..")
    data["u"] = mda.Universe(files["pdb"], files["dcd"])
    # data["rbd"] = data["u"].select_atoms("protein and chainID A and not name H*")
    # data["nb"] = data["u"].select_atoms("protein and chainID B and not name H*")
    data["rbd"] = data["u"].select_atoms("chainID A and  name CA")
    data["nb"] = data["u"].select_atoms("chainID B and name CA")
    
    data["t"] = mdt.load(files["dcd"], top = files["pdb"])
    data["t"].superpose(data["t"], 0) # align with 0th frame
    return data

# https://userguide.mdanalysis.org/1.0.0/examples/analysis/distances_and_contacts/contacts_native_fraction.html
def calcContacts(data, verbose = True):
    print("\nCalculating change in native contact frequency over trajectory..")
    rbdSel, nbSel = "chainID A and  name CA", "chainID B and  name CA"
    c = contacts.Contacts(data["u"],
                          select = (rbdSel, nbSel),
                          refgroup = (data["rbd"], data["nb"]),
                          radius = data["cutoff"],
                          method = "radius_cut")
    c.run(verbose = verbose)
    data["contacts"] = pd.DataFrame(c.results.timeseries,
                                    columns = ["Frame", "Contacts"])
    
    # Visualization
    plt.figure(figsize = (12, 9))
    data["contacts"].plot(x = "Frame")
    plt.ylabel("Native Contact Frequency")
    plt.title(f"Change in Native Contacts Frequency\n(Cutoff = {data['cutoff']} $\AA$)")
    plt.savefig(f"contacts{data['cutoff']}A.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return data

# https://userguide.mdanalysis.org/stable/examples/analysis/distances_and_contacts/distances_between_selections.html
# https://userguide.mdanalysis.org/stable/examples/analysis/distances_and_contacts/distances_within_selection.html
def calcDist(data, verbose = True):
    print("\nCalculating average pairwise distances..")
    rbdCom = data["rbd"].positions
    nbCom = data["nb"].positions

    dL = [ ]
    start = time.perf_counter()
    for frame in data["u"].trajectory:        
        d = distances.distance_array(nbCom, # reference (y)
                                     rbdCom) # # configuration (x)
        # need to learn how to make a python progress bar            
        dL.append(d)
    
    end = time.perf_counter()
    print(f"Finished calculating distances in {end - start} seconds")
    dA = np.array(dL)
    data["distance"] = np.mean(dA, axis=0) # average of d over frames. axis = 0 is average over depth/layers (frames)
    
    # Visualization
    # --- Inferno colormap ---
    inferno_cmap = colormaps["inferno"]

    fig0, ax0 = plt.subplots(figsize=(12,9))
    im0 = ax0.imshow(
        data["distance"],
        origin='upper',
        cmap=inferno_cmap
    )

    tick_interval = 10
    ax0.set_xticks(np.arange(0, len(rbdCom), tick_interval), minor=True)
    ax0.set_yticks(np.arange(0, len(nbCom), tick_interval), minor=True)

    plt.xlabel('RBD Residues')
    plt.ylabel('NB Residues')
    plt.title('Average Pairwise Distance Between NB and RBD Residues Over Trajectory')

    plt.xticks(rotation=90)

    # Colorbar
    cbar0 = fig0.colorbar(im0)
    cbar0.ax.set_ylabel("Distance ($\\AA$)")

    plt.savefig("distanceHeatmap.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Masking
    
    # Build a colormap that maps 0 → cutoff normally
    base_cmap = plt.cm.inferno
    new_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))

    # Set the "over" color for values > cutoff
    new_cmap.set_over(color="lightgray")

    fig1, ax1 = plt.subplots(figsize=(8,6))

    # --- BOTTOM LAYER: full distance map (viridis + lightgray) ---
    im1 = ax1.imshow(
        data["distance"],
        cmap=new_cmap,
        vmax=data["cutoff"],          # everything above cutoff uses "over" color
        interpolation='none',
        zorder=0
    )
    im1.set_clim(0, data["cutoff"])   # ensures consistent normalization

    # --- MIDDLE LAYER: CDR spans ---
    ax1.axhspan(31, 35, color='palevioletred', alpha=0.3, zorder=1, label='CDR1')
    ax1.axhspan(50, 66, color='mediumvioletred', alpha=0.3, zorder=1, label='CDR2')
    ax1.axhspan(97,108, color='blueviolet', alpha=0.3, zorder=1, label='CDR3')

    # --- TOP LAYER: highlight points ≤ cutoff ---
    highlight_mask = np.ma.masked_where(data["distance"] > data["cutoff"],
                                        data["distance"])

    im2 = ax1.imshow(
        highlight_mask,
        cmap="inferno",                    # highlight color map
        vmin=0,
        vmax=data["cutoff"],
        interpolation='none',
        zorder=2,
        alpha=0.9
    )

    # Colorbar for the base layer (0..cutoff, + "over")
    cbar1 = fig1.colorbar(im1, extend="max")
    cbar1.ax.set_ylabel('Distance ($\\AA$)')

    tick_interval = 10
    ax1.set_xticks(np.arange(0, len(rbdCom), tick_interval), minor=True)
    ax1.set_yticks(np.arange(0, len(nbCom), tick_interval), minor=True)
    plt.xlabel('RBD')
    plt.ylabel('NB')
    plt.title(f"Filtered Average Distance Between NB and RBD Residues\n(Cutoff = {data['cutoff']}$\\AA$)")
    plt.xticks(rotation=90)

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"filteredAvgD{data['cutoff']}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # --- NEW PLOT: only residues below cutoff (categorical axes) ---
    below_cutoff_mask = data["distance"] <= data["cutoff"]

    nb_residues_under = np.where(below_cutoff_mask.any(axis=1))[0]
    rbd_residues_under = np.where(below_cutoff_mask.any(axis=0))[0]

    if len(nb_residues_under) > 0 and len(rbd_residues_under) > 0:

        filtered_distance = data["distance"][np.ix_(nb_residues_under, rbd_residues_under)]

        # --- Mask for highlight layer ---
        highlight_mask = np.ma.masked_where(filtered_distance > data["cutoff"], filtered_distance)

        # --- Base colormap with neutral background ---
        base_cmap = plt.cm.inferno
        new_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
        new_cmap.set_over("lightgray")

        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # --- Bottom layer ---
        im0 = ax2.imshow(
            filtered_distance,
            origin='upper',
            cmap=new_cmap,
            vmax=data["cutoff"],
            interpolation='none',
            zorder=0
        )
        im0.set_clim(0, data["cutoff"])

        # --- ADD: thin grid lines at residue boundaries (behind everything) ---
        N_rbd = len(rbd_residues_under)
        N_nb  = len(nb_residues_under)
        ax2.set_xticks(np.arange(N_rbd) - 0.5, minor=True)
        ax2.set_yticks(np.arange(N_nb) - 0.5, minor=True)
        ax2.grid(which='minor', linestyle='-', linewidth=0.2, color='black', alpha=0.25)
        ax2.tick_params(which='minor', bottom=False, left=False)

        # --- CDR spans ---
        for start, end, color, label in [
            (31, 35, 'palevioletred', 'CDR1'),
            (50, 66, 'mediumvioletred', 'CDR2'),
            (97,108, 'blueviolet', 'CDR3')
        ]:
            overlap = np.intersect1d(np.arange(start, end+1), nb_residues_under)
            if len(overlap) > 0:
                y0 = np.where(nb_residues_under == overlap[0])[0][0]
                y1 = np.where(nb_residues_under == overlap[-1])[0][0]
                ax2.axhspan(y0, y1, color=color, alpha=0.3, label=label, zorder=1)

        # --- Highlight layer ---
        im1 = ax2.imshow(
            highlight_mask,
            origin='upper',
            cmap="inferno",
            vmin=0,
            vmax=data["cutoff"],
            interpolation='none',
            alpha=0.9,
            zorder=2
        )

        # --- Colorbar ---
        cbar = fig2.colorbar(im0, extend="max")
        cbar.ax.set_ylabel('Distance ($\\AA$)')

        # --- Tick spacing logic ---
        # N_rbd and N_nb already computed above

        # Show all ticks but shrink font depending on count
        xtick_font = 8 if N_rbd <= 40 else 6 if N_rbd <= 80 else 5
        ytick_font = 8 if N_nb  <= 40 else 6 if N_nb  <= 80 else 5

        ax2.set_xticks(np.arange(N_rbd))
        ax2.set_xticklabels([str(r) for r in rbd_residues_under],
                            rotation=90,
                            fontsize=xtick_font)

        ax2.set_yticks(np.arange(N_nb))
        ax2.set_yticklabels([str(r) for r in nb_residues_under],
                            fontsize=ytick_font)

        # Labels + Title
        plt.xlabel('RBD Residues')
        plt.ylabel('NB Residues')
        plt.title(f"Filtered Average Pairwise Distance Between Nb and Rbd Residues Over Trajectory\n (Cutoff = {data['cutoff']}$\\AA$)",
                pad=20)  # prevents title clipping

        # Make more room for rotated x-tick labels
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, left=0.20, top=0.92)

        plt.legend(loc='upper left', fontsize=8)

        # --- SAVE: centered in PNG and no clipping ---
        plt.savefig(
            f"belowCutoffDistanceColored{data['cutoff']}.png",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()
    
    return data

def calcContacts2(data, verbose = True):
    print("\nCalculating pairwise contacts frequency..")
    
    # May need
    # rbd = data["u"].select_atoms("protein and chainID A and not name H*")
    # nb = data["u"].select_atoms("protein and chainID B and not name H*")
    
    # rbdCom = data["rbd"].center_of_mass(compound = "residues")
    # nbCom = data["nb"].center_of_mass(compound = "residues")
    
    rbdCom = data["rbd"].positions
    nbCom = data["nb"].positions
    
    contactA = np.zeros((len(nbCom), len(rbdCom))) # initiailized contact matrix for each pairwise contact. Adds 1 each time distance is less than cutoff 
    start = time.perf_counter()
    for frame in data["u"].trajectory:        
        d = distances.distance_array(nbCom, # reference (y)
                                     rbdCom) # configuration (x)
        contactA += (d < data["cutoff"]) # add contact if less than cutoff
    
    end = time.perf_counter()
    print(f"Finished finding contacts in {end - start} seconds")
    
    # Normalize to frequency
    contactA = contactA / len(data["u"].trajectory)

    # Mask for highlighted contacts (≥50%)
    highlight_mask = np.ma.masked_where(contactA < 0.5, contactA)

    # --- Base colormap with neutral background ---
    base_cmap = plt.cm.inferno
    new_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    new_cmap.set_under("lightgray")   # values < 0.5 will appear neutral

    fig0, ax0 = plt.subplots(figsize=(8,6))

    # --- BOTTOM LAYER: full frequency heatmap (lightgray for <0.5) ---
    im0 = ax0.imshow(
        contactA,
        origin='upper',
        cmap=new_cmap,
        vmin=0.5,         # everything <0.5 uses “under” color
        vmax=1.0,
        interpolation='none',
        zorder=0
    )

    # --- MIDDLE LAYER: CDR spans ---
    ax0.axhspan(31, 35, color='palevioletred', alpha=0.3, zorder=1, label='CDR1')
    ax0.axhspan(50, 66, color='mediumvioletred', alpha=0.3, zorder=1, label='CDR2')
    ax0.axhspan(97,108, color='blueviolet', alpha=0.3, zorder=1, label='CDR3')

    # --- TOP LAYER: highlight contacts ≥ 0.5 ---
    im1 = ax0.imshow(
        highlight_mask,
        origin='upper',
        cmap="inferno",
        vmin=0.5,
        vmax=1.0,
        interpolation='none',
        zorder=2,
        alpha=0.9,
    )

    # --- Colorbar ---
    cbar0 = fig0.colorbar(im0, extend="min")  # show "under" arrow
    cbar0.ax.set_ylabel('Frequency')

    # --- Axes styling ---
    tick_interval = 10
    ax0.set_xticks(np.arange(len(rbdCom), minor=True)[::tick_interval])
    ax0.set_yticks(np.arange(len(nbCom), minor=True)[::tick_interval])

    plt.xlabel('RBD Residues')
    plt.ylabel('NB Residues')
    plt.title(f"Contact Frequency Between Nb and Rbd Residues\n(Cutoff = {data['cutoff']}$\\AA$)")
    plt.xticks(rotation=90)
    plt.legend(loc='upper left')

    plt.savefig(f"pairwiseContactFreq{data['cutoff']}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return data

def runPca(data, verbose = True):
    print("Running PCA..")
    atoms = data["t"].topology.select("name CA")
    coords = data["t"].xyz[:, atoms, :].reshape(len(data["t"]), -1) 
    # flatten to 2d matrix for PCA
    # flattened rows = frames: [x1 y1 z1 x2 y2 z2 ... xn yn zn]
    # columns = features
    
    nFeatures = coords.shape[1]
    coordsCentered = coords - coords.mean(axis=0) # centered matrix for PCA
    cov = (coordsCentered.T @ coordsCentered) / (nFeatures - 1)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    data["pc"] = coordsCentered@eigvecs # [frames x features] @ [features @ features]
    return data
    
def calcSfe(data, bins = 200, sigma = 2, temperature = 310):
    print("\nCalculating surface free energy..")
    kB = 1.380649E-23 # J/K
    T = 273.15 + temperature # K
    
    states, xedges, yedges = np.histogram2d(data["pc"][:,0],
                                       data["pc"][:,1],
                                       bins=bins)
    prob = states / np.sum(states) # probability of a state (bin)
    F = -kB*T*np.log(prob, where=(prob>0)) # absolute free energy
    deltaF = F - np.nanmin(F) # relative free energy, sets min to 0

    data["deltaFsmooth"] = gaussian_filter(deltaF, sigma = sigma)
    
    # Visualization
    plt.contourf(data["deltaFsmooth"].T, cmap='inferno')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="ΔF (kcal/mol)")
    plt.title("Surface Free Energy Plot")
    plt.savefig("sfePCA.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return data

# https://pubs.acs.org/doi/10.1021/acs.jcim.5c01725#fig1
def calcResCor(data):
    print("\nCalculating Residue Correlation..")
    nbN = len(data["nb"])
    rbdN = len(data["rbd"])
    nFrames = len(data["u"].trajectory)

    # Initialize empty arrays for position (x, y ,z) averages of residues
    nbAvgPos = np.zeros((nbN, 3))
    rbdAvgPos = np.zeros((rbdN, 3))

    for step in data["u"].trajectory:
        nbAvgPos += data["nb"].positions
        rbdAvgPos += data["rbd"].positions

    nbAvgPos /= nFrames
    rbdAvgPos /= nFrames

    # Compute inter-protein covariance
    # C(i,j) = < ΔR_i • ΔR_j >
    cov = np.zeros((nbN, rbdN))
    varNB  = np.zeros(nbN)
    varRBD = np.zeros(rbdN)

    # Centering the matrix
    for step in data["u"].trajectory:
        nbDelta = data["nb"].positions - nbAvgPos  # nb x 3
        rbdDelta = data["rbd"].positions - rbdAvgPos  # rbN x 3
        # Compute dot product for each pair (i,j)
        cov += nbDelta @ rbdDelta.T  # nb x rbN
        varNB  += np.sum(nbDelta**2,  axis=1)
        varRBD += np.sum(rbdDelta**2, axis=1)

    cov /= nFrames  # ensemble average
    varNB  /= nFrames
    varRBD /= nFrames

    # Compute normalized correlation
    # C(i,j) = c(i,j) / sqrt(c(i,i) * c(j,j))
    norm = np.zeros_like(cov)
    for i in range(nbN):
        for j in range(rbdN):
            denom = np.sqrt(varNB[i] * varRBD[j])
            norm[i, j] = cov[i, j] / denom if denom != 0 else 0

    # Apply distance cutoff for close contacts
    # Average distances between residues
    avgD = np.zeros((nbN, rbdN))
    for i in range(nbN):
        for j in range(rbdN):
            avgD[i, j] = np.linalg.norm(nbAvgPos[i] - rbdAvgPos[j])

    # Distance cutoff: 8 Å for positive correlations, 13 Å for negative correlations
    cutoffM = np.zeros_like(norm)
    cutoffM[(norm > 0) & (avgD <= 8)] = norm[(norm > 0) & (avgD <= 8)]
    cutoffM[(norm < 0) & (avgD <= 13)] = norm[(norm < 0) & (avgD <= 13)]
    
    data["attractionScore"] = np.sum(cutoffM[cutoffM > 0])
    data["repulsionScore"]  = np.sum(cutoffM[cutoffM < 0])
    data["bindingScore"]    = np.sum(cutoffM)
    
    # Visualization
    # --- Adaptive label sizing ---
    N_rbd = rbdN
    N_nb  = nbN

    xtick_font = 8 if N_rbd <= 40 else 6 if N_rbd <= 80 else 5
    ytick_font = 8 if N_nb  <= 40 else 6 if N_nb  <= 80 else 5

    fig_w = 10 if N_rbd <= 40 else 12 if N_rbd <= 80 else 14
    fig_h = 8 if N_nb  <= 40 else 10 if N_nb  <= 80 else 12

    fig0, ax0 = plt.subplots(figsize=(fig_w, fig_h))

    # --- Create masked array for heatmap points ---
    masked_cutoffM = np.ma.masked_where(cutoffM == 0, cutoffM)  # only show points with data

    # --- Neutral background (light gray) ---
    neutral_bg = np.ones_like(cutoffM)
    cmap_bg = ListedColormap(['lightgray'])
    ax0.imshow(neutral_bg, origin='upper', cmap=cmap_bg, interpolation='none', zorder=0)

    # --- CDR shading bands (above background, below heatmap points) ---
    for start, end, color, label in [
        (31, 35, 'palevioletred', 'CDR1'),
        (50, 66, 'mediumvioletred', 'CDR2'),
        (97,108,'blueviolet', 'CDR3')
    ]:
        ax0.axhspan(start, end, color=color, alpha=0.3, zorder=1, label=label)

    # --- Main heatmap points (above bands) ---
    im = ax0.imshow(masked_cutoffM, cmap="coolwarm", vmin=-1, vmax=1,
                    origin='upper', interpolation='none', zorder=2)

    # --- Thin gridlines behind everything ---
    ax0.set_xticks(np.arange(N_rbd) - 0.5, minor=True)
    ax0.set_yticks(np.arange(N_nb) - 0.5, minor=True)
    ax0.grid(which='minor', linestyle='-', linewidth=0.2, color='black', alpha=0.25)
    ax0.tick_params(which='minor', bottom=False, left=False)

    # --- Residue labels ---
    tick_interval = 10
    ax0.set_xticks(np.arange(0, N_rbd, tick_interval))
    ax0.set_yticks(np.arange(0, N_nb, tick_interval))
    ax0.set_xticklabels([str(i) for i in range(0, N_rbd, tick_interval)],
                        rotation=90, fontsize=xtick_font)
    ax0.set_yticklabels([str(i) for i in range(0, N_nb, tick_interval)],
                        fontsize=ytick_font)

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax0)
    cbar.set_label("Correlation")

    # --- Labels + Title ---
    ax0.set_xlabel("RBD Residues")
    ax0.set_ylabel("NB Residues")
    ax0.set_title("Close-Contact Inter-Protein Covariance Matrix", pad=20)

    # --- Legend ---
    ax0.legend(loc='upper left', fontsize=8)

    # --- Save PNG centered ---
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, left=0.20, top=0.92)
    plt.savefig("residueCorrelation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # FILTERED GRAPH
    # --- Adaptive label sizing ---
    N_rbd = rbdN
    N_nb  = nbN

    xtick_font = 8 if N_rbd <= 40 else 6 if N_rbd <= 80 else 5
    ytick_font = 8 if N_nb  <= 40 else 6 if N_nb  <= 80 else 5

    fig_w = 10 if N_rbd <= 40 else 12 if N_rbd <= 80 else 14
    fig_h = 8 if N_nb  <= 40 else 10 if N_nb  <= 80 else 12

    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    # --- Create masked array for heatmap points ---
    masked_cutoffM = np.ma.masked_where(cutoffM == 0, cutoffM)  # only show points with data
    neutral_bg = np.ma.masked_where(cutoffM != 0, cutoffM)      # neutral background for empty spots

    # --- Neutral background (light gray) ---
    cmap_bg = ListedColormap(['lightgray'])
    ax1.imshow(neutral_bg, origin='upper', cmap=cmap_bg, interpolation='none', zorder=0)

    # --- CDR shading bands (above background, below points) ---
    ax1.axhspan(31, 35, color='palevioletred', alpha=0.3, zorder=1, label='CDR1')
    ax1.axhspan(50, 66, color='mediumvioletred', alpha=0.3, zorder=1, label='CDR2')
    ax1.axhspan(97,108, color='blueviolet', alpha=0.3, zorder=1, label='CDR3')
    
    # --- Main heatmap points ---
    im = ax1.imshow(masked_cutoffM, cmap="coolwarm", vmin=-1, vmax=1,
                    origin='upper', interpolation='none', zorder=2)

    # --- Thin gridlines behind everything ---
    ax1.set_xticks(np.arange(N_rbd) - 0.5, minor=True)
    ax1.set_yticks(np.arange(N_nb) - 0.5, minor=True)
    ax1.grid(which='minor', linestyle='-', linewidth=0.2, color='black', alpha=0.25)
    ax1.tick_params(which='minor', bottom=False, left=False)

    # --- Residue labels ---
    ax1.set_xticks(np.arange(N_rbd))
    ax1.set_yticks(np.arange(N_nb))
    ax1.set_xticklabels([str(i) for i in range(N_rbd)], rotation=90, fontsize=xtick_font)
    ax1.set_yticklabels([str(i) for i in range(N_nb)], fontsize=ytick_font)

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Correlation")

    # --- Labels + Title ---
    ax1.set_xlabel("RBD Residues")
    ax1.set_ylabel("NB Residues")
    ax1.set_title("Filtered Close-Contact Inter-Protein Covariance Matrix", pad=20)

    # --- Layout adjustments ---
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, left=0.20, top=0.92)

    # --- Legend ---
    plt.legend(loc='upper left', fontsize=8)

    # --- Save PNG centered ---
    plt.savefig("residueCorrelationFiltered.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return data

def runUmap(data, n_neighbors=90, min_dist=0.6, n_components=3):
    print("\nRunning UMAP..")
    start = time.perf_counter()
    pcs = data["pc"][:, :10]   # first 10 PCs
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        random_state=42
    )
    
    emb = reducer.fit_transform(pcs)
    data["umap"] = emb
    
    end = time.perf_counter()
    print(f"Finished finding contacts in {end - start} seconds")
    
    # Graphing
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], c=np.arange(len(emb)), cmap="inferno", s=3)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Frame")
    plt.title("UMAP of PCA Projection")
    plt.savefig(f"umapN{n_neighbors}D{min_dist}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return data
    
#     return data
def clusterUmap(data, min_cluster_size=500):
    print("\nClustering UMAP with no noise..")
    emb = data["umap"]

    # Step 1: HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_epsilon=0.01
    )
    labels = clusterer.fit_predict(emb)

    # Step 2: Assign noise points (-1) to nearest cluster
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        unique_labels = np.unique(labels[labels != -1])
        centroids = np.array([emb[labels == lbl].mean(axis=0) for lbl in unique_labels])

        for i in noise_idx:
            nearest = np.argmin(cdist([emb[i]], centroids))
            labels[i] = unique_labels[nearest]

    data["cluster"] = labels

    # Step 3: Plot with legend instead of colorbar
    plt.figure(figsize=(8,6))
    unique_clusters = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for idx, cluster_id in enumerate(unique_clusters):
        cluster_points = emb[labels == cluster_id]
        plt.scatter(
            cluster_points[:,0], 
            cluster_points[:,1], 
            color=cmap(idx % 10), 
            s=4, 
            label=f"Cluster {cluster_id}"
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Clustered UMAP")
    plt.legend(markerscale=3, fontsize=10)
    plt.savefig("umapClustered.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return data


def calcRmsd(data, verbose = True):
    rFirst = rms.RMSD(data["nb"],
                 data["nb"],
                 select = "name CA",
                 ref_frame = 0).run(verbose = verbose)
    rFirstM = rFirst.results.rmsd.T
    
    rLast = rms.RMSD(data["nb"],
                 data["nb"],
                 select = "name CA",
                 ref_frame = -1).run(verbose = verbose)
    rLastM = rLast.results.rmsd.T
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8,6))

    inferno_cmap = colormaps["inferno"]

    ax.plot(rFirstM[1], rFirstM[2], color=inferno_cmap(0.3), label="RMSD vs First Frame")
    ax.plot(rLastM[1],  rLastM[2],  color=inferno_cmap(0.7), label="RMSD vs Last Frame")

    ax.set_xlabel("Frame")
    ax.set_ylabel("RMSD ($\AA$)")
    ax.set_title("RMSD Over Trajectory")
    ax.legend()
    plt.savefig("rmsd.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return data


### RUN ####

def runAll(pdb = "system.pdb", 
           dcd = "trajectory.dcd"):
    data = {"cutoff": 13}
    files = {
        "pdb": pdb,
        "dcd": dcd
    }
    
    data = makeU(files, data)
    data = calcDist(data)
    data = calcContacts(data)
    # data = calcContacts2(data)
    
    data = calcResCor(data)
    data = runPca(data)
    data = calcSfe(data)
    data = calcRmsd(data)
    data = runUmap(data)
    data = clusterUmap(data) 
    
    with open("binding.txt", "w") as f:
        f.write(f"attractionScore   {data['attractionScore']}\n")
        f.write(f"repulsionScore    {data['repulsionScore']}\n")
        f.write(f"bindingScore    {data['bindingScore']}")

    return None

if __name__ == "__main__":
    # For running once
    # cwDir = os.getcwd()
    # filesDir = "/nbD7Run24Nov25"
    # os.chdir(cwDir + filesDir)
    runAll()
    
    
#### NOTES ####

# Fast distances caluclations: https://docs.mdanalysis.org/stable/documentation_pages/lib/distances.html#MDAnalysis.lib.distances.distance_array


# Original umap function including noise
# def clusterUmap(data, min_cluster_size=500):
#     print("Clustering UMAP..")
#     emb = data["umap"]
    
#     # HDBSCAN = best for UMAP landscapes
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=min_cluster_size,
#         min_samples=1,
#         cluster_selection_epsilon=0.01
#     )
    
#     labels = clusterer.fit_predict(emb)
#     data["cluster"] = labels

#     # Plot
#     plt.figure(figsize=(8,6))
#     plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="tab10", s=4)
#     plt.xlabel("UMAP1")
#     plt.ylabel("UMAP2")
#     plt.title("Clustered UMAP (HDBSCAN)")
#     plt.colorbar(label="Cluster ID")
#     plt.savefig("clustered_umap.png", dpi=300)
#     plt.close()

# def analyze_binding_modes(files, data):
#     # ---- Load + Align ----
#     traj = mdt.load(files["dcd"], top=files["pdb"])
#     traj.superpose(traj, 0)

#     # ---- PCA ----
#     atoms = traj.topology.select("name CA")
#     coords = traj.xyz[:, atoms, :].reshape(len(traj), -1)
#     coords_centered = coords - coords.mean(axis=0)

#     cov = coords_centered.T @ coords_centered / (coords_centered.shape[1] - 1)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     idx = np.argsort(eigvals)[::-1]
#     eigvecs = eigvecs[:, idx]

#     pcs = coords_centered @ eigvecs
#     pc1, pc2 = pcs[:, 0], pcs[:, 1]

#     # ---- KMeans: identify PCA basins ----
#     kmeans = KMeans(n_clusters=3, random_state=0)
#     labels = kmeans.fit_predict(pcs[:, :2])

#     data["pc"] = pcs
#     data["cluster"] = labels

#     # ---- Plot PCA clusters ----
#     plt.figure()
#     plt.scatter(pc1, pc2, c=labels, cmap="viridis", s=2)
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.colorbar(label="Cluster")
#     plt.savefig("PCA_clusters.png")
#     plt.close()

#     # ---- Contact calculation setup ----
#     u = data["u"]
#     rbd_com = data["rbd"].center_of_mass(compound="residues")
#     nb_com = data["nb"].center_of_mass(compound="residues")

#     n_nb = len(nb_com)
#     n_rbd = len(rbd_com)
#     cutoff = data["cutoff"]

#     # ---- One contact map per cluster ----
#     contact_maps = {c: np.zeros((n_nb, n_rbd)) for c in range(3)}
#     counts = {c: 0 for c in range(3)}

#     for i, ts in enumerate(u.trajectory):
#         c = labels[i]  # cluster of this frame
        
#         d = distances.distance_array(nb_com, rbd_com, box=u.dimensions)
#         contact_maps[c] += (d < cutoff).astype(float)
#         counts[c] += 1

#     # ---- Normalize + Plot ----
#     for c in range(3):
#         cm = contact_maps[c] / counts[c]

#         plt.figure(figsize=(8, 6))
#         plt.imshow(cm, origin='upper', cmap='viridis', vmin=0, vmax=1)
#         plt.title(f"Cluster {c} Contact Map")
#         plt.xlabel("RBD residues")
#         plt.ylabel("NB residues")
#         plt.colorbar(label="Contact Frequency")
#         plt.savefig(f"cluster_{c}_contacts.png")
#         plt.close()

#     return data





# UMAP 90 and 0.6 worked for NB1