import h5py
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import cmasher as cmr
cmap = plt.get_cmap('cmr.lavender')

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
regions = (['00','01','02','03','04','05','06','07','08'] +
           [str(i) for i in np.arange(10, 40)])
snapshot = '008_z007p000'
sps_model = 'bpass'

# Get the matches.
best_matches = ['04:4', '10:125']
colours = ['green', 'orange']
markers = ['o', 's']
others = ['00:361', '08:503', '04:14', '15:432', '06:755']
identifiers = best_matches + others

# Set up the plot.
fig, ax = plt.subplots(figsize=(3.78, 3.78))

bh_masses = []
s_masses = []
ssfrs = []
with h5py.File(master_path) as master:

    # For each region.
    for region in regions:
        if region == '38':
            continue
        key = f'{region}/{snapshot}'

        # Get indices from the photometry sample.
        cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{region}_{snapshot}_{sps_model}.hdf5'
        with h5py.File(cat_path) as cat:
            indices = cat['Galaxies/MasterRegionIndex'][:]
            indices = indices[np.argsort(indices)]

        # Store the sSFRs.
        ssfr = np.log10(1e9 * master[f'{key}/Galaxy/SFR_aperture/30/50Myr'][indices] / 
                        (master[f'{key}/Galaxy/Mstar_aperture/30'][indices] * 1e10))
        ssfrs += list(ssfr)

        bh_length = master[f"{key}/Galaxy/BH_Length"][:]
        bh_start = np.insert(np.cumsum(bh_length), 0, 0)    

        # Get the stellar and BH masses.
        for idx in indices:

            s_mass = master[f'{key}/Galaxy/Mstar_aperture/30'][idx] * 1e10
            bh_mass = np.sum(master[f"{key}/Particle/BH_Mass"][bh_start[idx]:bh_start[idx+1]] * 1e10)
            s_masses.append(s_mass)
            bh_masses.append(bh_mass)

            if f'{region}:{idx}' in identifiers:
                id = f'{region}:{idx}'
                i = identifiers.index(id)

                if id in best_matches:
                    ax.scatter(np.log10(s_mass), np.log10(bh_mass / s_mass), color=colours[i], 
                            label=f'FRA-{i+1}', s=40, zorder=2)
                else:
                    ax.scatter(np.log10(s_mass), np.log10(bh_mass / s_mass), color='red', 
                               marker='x', s=20, zorder=2)

# If there are any 'other' matches, add a legend entry for them.              
if len(others) > 0:
    ax.scatter(-99, -99, color='red', s=20, label='FRA-W', marker='x')

# Plot all FLARES galaxies coloured by sSFR.
bh_masses = np.array(bh_masses)
s_masses = np.array(s_masses)
ssfrs = np.array(ssfrs)

sc = ax.scatter(np.log10(s_masses), np.log10(bh_masses/s_masses), c=ssfrs, alpha=1, s=2, zorder=1, cmap=cmap)

# Tidy up.
ax.set_ylim(-4.8, -1.8)
ax.set_xlim(8.5, 10.8)
ax.set_xlabel('$\\log_{10}(\\mathrm{M_{\\ast}} \ / \ \\mathrm{M_{\\odot}})$')
ax.set_ylabel('$\\log_{10}(\\mathrm{M_{\\bullet}} \ / \ \\mathrm{M_{\\ast}})$')

ax.legend(ncol=1, frameon=True, framealpha=1, facecolor='white', fancybox=False, 
          edgecolor='grey', fontsize=8)

pos = ax[0].get_position()
cbar_height = 0.02
cbar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, cbar_height])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label("$\\log_{10}(\\mathrm{sSFR}_{50} \\ \ / \\ \\mathrm{Gyr}^{-1})$", labelpad=7)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(size=2)
cb.ax.minorticks_off()

fig.savefig("plots/bh_mass_ratio.pdf", dpi=300)
plt.show()
