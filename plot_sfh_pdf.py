import h5py
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck13 as cosmo

import matplotlib.pyplot as plt
import cmasher as cmr
plt.style.use('style.mplstyle')
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.labeltop'] = False
cmap = plt.get_cmap('cmr.lavender')

# Redshift dependent agebin function.
def zred_to_agebins(zred=None, agebins=None, nbins_sfh=None, **extras):
    tuniv = cosmo.age(zred).value*1e9
    agelims = np.concatenate(([0], np.log10([0.01e9,0.05e9]), 
                              np.log10(np.linspace(0.1e9, tuniv, nbins_sfh-2))))
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

# Get the RQG bin info.
edges = 10 ** zred_to_agebins(7.29, nbins_sfh=8)/1e6
rqg_bins = np.concatenate([edges[:, 0], edges[-1:, 1]])
dt = np.diff(rqg_bins)
x = np.repeat(rqg_bins, 2)[1:-1]

# FLARES relevant info.
master_path = "flares.hdf5"
rqg_path = 'rqg_sfh.fits'
snapshot = "008_z007p000"

# Get the matches.
best_matches = ['04:4', '10:125']
colours = ['green', 'orange']
others = []
identifiers = best_matches + others

# Set up the plot
fig, ax = plt.subplots(figsize=(3.78, 3.78))

# For each match.``
with h5py.File(master_path, "r") as hdf:

    for i, id in enumerate(identifiers):
        reg, idx = id.split(':')
        idx = int(idx)
        key = f"{reg}/{snapshot}/"

        # Get the particle stellar masses and ages
        slength = hdf[f"{key}/Galaxy/S_Length"][:]
        sstart = np.cumsum(slength)
        sstart = np.insert(sstart, 0, 0)

        smass = hdf[f"{key}/Particle/S_MassInitial"][sstart[idx] : sstart[idx + 1]] * 1e10
        sage = hdf[f"{key}/Particle/S_Age"][sstart[idx] : sstart[idx + 1]] * 1e3

        # The mass formed within each bin.
        H, _ = np.histogram(sage, bins=rqg_bins, weights=smass)

        # Divide by time interval to get SFR.
        sfr = H / dt
        y = np.repeat(sfr / np.sum(H), 2)

        # Plot with steps.
        if id in best_matches:
            ax.plot(x, y, color=colours[i], alpha=1, label=f'FRA-{i+1}', 
                    zorder=1-((0.25/len(best_matches))*(i+1)))
        else:
            ax.plot(x, y, color='red', alpha=0.5, linestyle='--', zorder=0.1)

# If there are any 'other' matches, add a legend entry for them.
if len(others) > 0:
    ax.plot([-99, -99], [-99, -99], color='red', alpha=0.5, linestyle='--', label='FRA-W')

# Load in the RQG SFHs.
rqg_data = Table.read(rqg_path)

# Normalise and plot.
fiducial = np.column_stack((rqg_data['SFR_fiducial_lo'], rqg_data['SFR_fiducial'], 
                            rqg_data['SFR_fiducial_hi'])) * 1e6
mass_formed = fiducial * dt[:, np.newaxis]
fiducial /= mass_formed[:, 1].sum()

ax.plot(x, np.repeat(fiducial[:, 1], 2), color = 'black', label='Fiducial', 
        linewidth=2, zorder=0.5)
ax.fill_between(x, np.repeat(fiducial[:, 0], 2), np.repeat(fiducial[:, 2], 2), color='black', 
                alpha=0.2, zorder=0.5)

highz = np.column_stack((rqg_data['SFR_highz_lo'], rqg_data['SFR_highz'], 
                         rqg_data['SFR_highz_hi'])) * 1e6
mass_formed = highz * dt[:, np.newaxis]
highz /= mass_formed[:, 1].sum()

ax.plot(x, np.repeat(highz[:, 1], 2), color='saddlebrown', label='High-Z', linewidth=2, zorder=0.75)
ax.fill_between(x, np.repeat(highz[:, 0], 2), np.repeat(highz[:, 2], 2), color='saddlebrown', 
                alpha=0.2, zorder=0.75)

# Tidy up.
ax.invert_xaxis()

ax.legend(loc=2, ncol=1, frameon=True, framealpha=1, facecolor='white',
                fancybox=False, edgecolor='grey', fontsize=8)

ax.set_ylabel("$\\mathrm{SFR} \ / \ \\mathrm{M_{\\ast}} \ / \ \\delta_{t}$")
ax.set_ylabel("$\\frac{\\mathrm{SFR}}{\\mathrm{M_{\\ast}}} \ / \ \\mathrm{Myr^{-1}}$")
ax.set_xlabel("$t_{\\mathrm{LB}} \ / \ \\mathrm{Myr}$")

ax.set_ylim(1e-5, 0.0085)
ax.set_xlim(600, 0)

top_ax = ax.secondary_xaxis('top')
top_ax.set_xlabel("$\\mathrm{Redshift}$", labelpad=7)

labels = [20, 15, 12, 10, 9, 8, 7.29]
labels_ = [f'${lab}$' for lab in labels]
positions = [(cosmo.age(7.29)-cosmo.age(label)).value*1e3 for label in labels]
top_ax.set_xticks(positions)
top_ax.set_xticklabels(labels_) 
top_ax.tick_params(which='minor', top=False, labelsize=8)

ax.tick_params(which='both', top=False)

fig.savefig("plots/sfh_pdf.pdf", bbox_inches="tight", dpi=300)
#plt.show()
