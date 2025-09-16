import h5py
import numpy as np
from astropy.cosmology import Planck13 as cosmo

import matplotlib.pyplot as plt
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.labeltop'] = False
plt.style.use('/cosma/home/dp276/dc-turn11/style.mplstyle')

# Redshift dependent agebin function.
def zred_to_agebins(zred=None, agebins=None, nbins_sfh=None, **extras):
    tuniv = cosmo.age(zred).value*1e9
    agelims = np.concatenate(([0], np.log10([0.01e9,0.05e9]), 
                              np.log10(np.linspace(0.1e9, tuniv, nbins_sfh-2))))
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

# Get the RQG bins.
edges = 10 ** zred_to_agebins(7, nbins_sfh=8)/1e6
rqg_bins = np.concatenate([edges[:, 0], edges[-1:, 1]])
# Widths in yr.
dt = np.diff(rqg_bins)*1e6

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
snapshot = "008_z007p000"
identifiers = ['04:4', '10:125']
colours = ['green', 'orange']
sps_models = {'fsps': 'red', 'bpass': 'blue', 'bpass_alpha': 'purple'}

# Set up the plot
fig, ax = plt.subplots(nrows=len(identifiers), ncols=1, figsize=(3.78, 2.5 * len(identifiers)), 
                         sharex=True)
fig.subplots_adjust(hspace=0.03)
if len(identifiers) == 1:
    ax = [ax]

with h5py.File(master_path, "r") as master:

    # For each FLARES match.
    for i, id in enumerate(identifiers):

        reg, idx = id.split(':')
        idx = int(idx)
        key = f"{reg}/{snapshot}"

        # Get the particle stellar masses and ages.
        slength = master[f"{key}/Galaxy/S_Length"][:]
        sstart = np.cumsum(slength)
        sstart = np.insert(sstart, 0, 0)

        smass = master[f"{key}/Particle/S_Mass"][sstart[idx] : sstart[idx + 1]] * 1e10
        sage = master[f"{key}/Particle/S_Age"][sstart[idx] : sstart[idx + 1]] * 1e3

        # Calculate the SFH.
        H, _ = np.histogram(sage, bins=rqg_bins, weights=smass)
        H /= dt

        # Plot as steps.
        x = np.repeat(rqg_bins, 2)[1:-1] 
        y = np.repeat(H, 2)       
        ax[i].plot(x, y, color=colours[i], label = f'FRA-{i+1}', linewidth=2, zorder=0)

        # Plot SFH from each SPS model.
        for sps_model, colour in sps_models.items():
            result_file = f'{base_path}/data/prospector_outputs/flares_prospector_{reg}_{idx}_{sps_model}_fiducial_dict.npy'
            results = np.load(result_file, allow_pickle=True).item()

            label = sps_model.upper().replace('_ALPHA', '-$\\alpha$')
            ax[i].plot(results['t_lb']*1e3, np.median(results['sfh'], axis=0), color=colour, 
                       alpha=0.8, label=label, zorder=1)

        # Reverse the x-axes.
        ax[i].invert_xaxis()

        # Tidy up.
        ax[i].legend(ncol=1, frameon=True, framealpha=1, facecolor='white',
                     fancybox=False, edgecolor='grey', fontsize=8)
        
        ax[i].set_ylabel("$\\mathrm{SFR} \ / \ \\mathrm{M_{\\odot}yr^{-1}}$")
        ax[i].set_ylim(0.0001)
        if i == len(identifiers) - 1:
            ax[i].set_xlabel("$t_{\\mathrm{LB}} \ / \  \\mathrm{Myr}$")
            ax[i].set_xlim(650, 0)

        top_ax = ax[i].secondary_xaxis('top')
        labels = [20, 15, 12, 10, 9, 8, 7.29]
        positions = [(cosmo.age(7)-cosmo.age(label)).value*1e3 for label in labels]
        top_ax.set_xticks(positions)
        top_ax.tick_params(which='minor', top=False)
        if i == 0:
            top_ax.set_xlabel("Redshift", labelpad=7)
            top_ax.set_xticklabels(labels)
        else:
            top_ax.set_xticklabels([])

        ax[i].tick_params(which='both', top=False)

fig.savefig("plots/prospector_sfhs.pdf", bbox_inches="tight", dpi=300)
plt.show()

