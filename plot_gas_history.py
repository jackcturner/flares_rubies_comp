import re
import h5py
import numpy as np
from utils import get_progenitors

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use('style.mplstyle')

import cmasher as cmr
cmap = plt.get_cmap("cmr.guppy_r")

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
snapshot = '008_z007p000'
regions = (['00','01','02','03','04','05','06','07', '08','09'] + 
           [str(i) for i in np.arange(10, 40)])

# Will find the main progenitors in these snapshots.
progen_snaps = ['001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', 
                '005_z010p000', '006_z009p000', '007_z008p000']

# Get the redshift of each snapshot for the colour bar.
zs = []
for snap in progen_snaps + [snapshot]:
    match = re.search(r'_z(\d{3})p(\d{3})', snap)
    redshift = float(f"{match.group(1)}.{match.group(2)}")
    zs.append(redshift)
z = np.array(zs)
norm = mcolors.BoundaryNorm(boundaries=np.arange(min(z)-0.5, max(z)+0.6, 1), ncolors=cmap.N)

# Get the matches.
identifiers = ['04:4', '10:125']
colours = ['green', 'orange']
markers = ['o', 's']

# Set up the plot.
fig, ax = plt.subplots(1, 1, figsize=(3.78, 3.78), sharex=True)
fig.subplots_adjust(hspace=0.03)

with h5py.File(master_path, 'r') as master:

    g_fracs = []
    mstars = []

    # For each region.
    for region in regions:
        if region == '38':
            continue
        key = f'{region}/{snapshot}'

        # Get indices from the photometry sample.
        with h5py.File(f'{base_path}/data/{snapshot}/RUBIES_COMP_{region}_{snapshot}.hdf5') as cat:
            indices = cat['Galaxies/MasterRegionIndex'][:]

        # Calculate the gas fraction in each object.
        for idx in indices:

            total_mass = ((master[f'{key}/Galaxy/Mstar'][idx] * 1e10) + 
                          (master[f'{key}/Galaxy/Mgas'][idx] * 1e10))

            g_fracs.append((master[f'{key}/Galaxy/Mgas'][idx] * 1e10) / total_mass)
            mstars.append(np.log10(master[f'{key}/Galaxy/Mstar'][idx] * 1e10))

            # Plot a track with redshift for the matches.
            if f'{region}:{idx}' in identifiers:
                id = f'{region}:{idx}'
                i = identifiers.index(id)

                # Identify the progenitors and decendents.
                match_indices, matches = get_progenitors(region, idx, snapshot, progen_snaps,
                                                         master_path)
                # Add original snapshot.
                match_indices.append(idx)
                matches.append(1e6)
                matches = np.array(matches)

                # Get the gas fraction for each one.
                progen_fracs = []
                progen_m = []
                for match_idx, snap in zip(match_indices, progen_snaps + [snapshot]):
                
                    key_ = f'{region}/{snap}'

                    total_mass = ((master[f'{key_}/Galaxy/Mstar'][match_idx] * 1e10) + 
                                  (master[f'{key_}/Galaxy/Mgas'][match_idx] * 1e10))
                    g_frac = (master[f'{key_}/Galaxy/Mgas'][match_idx] * 1e10) / total_mass

                    progen_fracs.append(g_frac)
                    progen_m.append(np.log10(master[f'{key_}/Galaxy/Mstar'][match_idx] * 1e10))

                # Only include reliable matches in the track.
                s = matches > 100
                progen_m = np.array(progen_m)
                progen_fracs = np.array(progen_fracs)

                ax.plot(progen_m[s], progen_fracs[s], color=colours[i], label=f'FRA-{i+1}', 
                        marker=markers[i], markersize=5.5, zorder=1.5)
                sc = plt.scatter(progen_m[s], progen_fracs[s], c=z[s], cmap=cmap, norm=norm, 
                                 s=40, marker=markers[i], zorder=1.6)

# Plot the FLARES points.
g_fracs = np.array(g_fracs)
mstars = np.array(mstars)
ax.scatter(mstars, g_fracs, s=0.05, color='grey', zorder=1, marker='o')

# Tidy up.
ax.legend(frameon=True, framealpha=1, ncols=1, facecolor='white', 
            fancybox=False, edgecolor='grey', fontsize=8, loc='lower left')

ax.set_xlim(8.5, 10.8)
ax.set_ylim(-0.05, 1.05)

ax.set_ylabel('$\\mathrm{M_{gas}} \ / \ (\\mathrm{M_{gas}}+\\mathrm{M_{\\ast}})$')
ax.set_xlabel('$\\log_{10}(\\mathrm{M}_{\\mathrm{\\ast}} \ / \ \\mathrm{M}_{\\odot})$')

pos = ax.get_position()
cbar_height = 0.02
cbar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, cbar_height])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', ticks=z)
cb.set_label("$\\mathrm{Redshift}$", labelpad=7)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(size=2)
cb.ax.minorticks_off()
cb.ax.invert_xaxis()

fig.savefig('plots/gas_history.pdf', dpi=300, bbox_inches='tight')
plt.show()
