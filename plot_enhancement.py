import h5py
import numpy as np
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import cmasher as cmr
cmap = plt.get_cmap('cmr.lavender')

# FLARES relevant info.
base_path = './'
master_path = 'flares_abundances.hdf5'
regions = (['00','01','02','03','04','05','06','07','08','09'] + 
           [str(i) for i in np.arange(10, 40)])
snapshot = '008_z007p000'
sps_model = 'bpass'

# Get the matches.
best_matches = ['04:4', '10:125']
colours = ['green', 'orange']
markers = ['o', 's']
others = ['00:361', '08:503', '04:14', '15:432', '06:755']
identifiers = best_matches + others

# Fe and O atomic masses, solar O/Fe ratio.
fe_mass = 55.845
o_mass = 15.999
solar_alpha = 1.23 

# Set up the plot.
fig, ax = plt.subplots(1, 1, figsize=(3.78,3.78))

fe = []
o = []
masses = []
Zs = []
with h5py.File(master_path) as master:

    for reg in regions:
        if reg == '38':
            continue
        key = f'{reg}/{snapshot}'

        # Get indices from the photometry sample.
        cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{reg}_{snapshot}_{sps_model}.hdf5'
        with h5py.File(cat_path, 'r') as cat:
            idxs = cat['Galaxies/MasterRegionIndex'][:]
            idxs = idxs[np.argsort(idxs)]

        # Start calculating enhancement and store properties.
        fe += list(master[f'{key}/Galaxy/Abundances/S_Iron'][idxs] * o_mass)
        o += list(master[f'{key}/Galaxy/Abundances/S_Oxygen'][idxs] * fe_mass)

        masses += list(np.log10(master[f'{key}/Galaxy/Mstar_aperture/30'][idxs] * 1e10))
        Zs += list(np.log10(master[f'{key}/Galaxy/Metallicity/MassWeightedStellarZ'][idxs]/0.019))

        # Highlight the matches.
        for idx in idxs:
            if f'{reg}:{idx}' in identifiers:
                id = f'{reg}:{idx}'
                i = identifiers.index(id)

                mass = master[f'{key}/Galaxy/Mstar_aperture/30'][idx] * 1e10
                alpha = np.log10((master[f'{key}/Galaxy/Abundances/S_Oxygen'][idx] * fe_mass) /
                                 (master[f'{key}/Galaxy/Abundances/S_Iron'][idx] * o_mass)) - solar_alpha
                
                if id in best_matches:
                    ax.scatter(np.log10(mass), alpha, color=colours[i], zorder=1.5, 
                               label=f'FRA-{i+1}', s=40)
                else:
                    ax.scatter(np.log10(mass), alpha, color='red', marker='x', zorder=1, s=20)

# If there are any 'other' matches, add a legend entry for them.
if len(others) > 0:
    ax.scatter(-99, -99, color='red', s=20, label='FRA-W', marker='x')

# Plot the reamining FLARES galaxies.
fe = np.array(fe)
o = np.array(o)
masses = np.array(masses)

alpha = np.log10((o)/(fe)) - solar_alpha
sc = ax.scatter(masses, alpha, c=Zs, s=1, cmap=cmap, zorder=0)

# Add a running median.
x = np.linspace(masses.min()*0.99, masses.max()*1.01, 100)
bins = np.linspace(masses.min(), masses.max(), 20)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
median_alpha, _, _ = binned_statistic(masses, alpha, statistic='median', bins=bins)

ax.plot(bin_centers, median_alpha, linewidth=3, color='grey', alpha=0.8)

# Tidy up.
ax.legend(frameon=True, framealpha=1, ncols=1, facecolor='white', 
          fancybox=False, edgecolor='grey', fontsize=8)

ax.set_xlim(8.5, 10.8)
ax.set_ylim(0.62, 0.85)

ax.set_xlabel('$\\log_{10}(\\mathrm{M}_{\\ast} \ / \ \\mathrm{M}_{\\odot})$')
ax.set_ylabel('$\\log_{10}(\\mathrm{O/Fe}) - \\log_{10}(\\mathrm{O/Fe})_{\\odot}$')

pos = ax.get_position()
cbar_height = 0.0175
cbar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, cbar_height])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label("$\\log_{10}(\\mathrm{Z}_{\\ast} \\, / \\, \\mathrm{Z}_{\\odot})$", labelpad=7)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(size=2)
cb.ax.minorticks_off()

fig.savefig('plots/alpha_enhancement.pdf', dpi=300, bbox_inches='tight')
plt.show()