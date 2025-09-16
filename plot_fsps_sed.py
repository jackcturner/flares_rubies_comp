import h5py
import numpy as np
from utils import normalise_seds

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

# FLARES relevant info.
base_path = './'
snapshot = '008_z007p290'
regions = (['00','01','02','03','04','05','06','07','08','09'] + 
           [str(i) for i in np.arange(10, 40)])
sps_model = 'fsps'

# Get the matches.
best_matches = ['04:4', '10:125', '04:14']
colours = ['green', 'orange', 'yellow']
others = ['14:715', '08:503', '15:432']
identifiers = best_matches + others

# Only show filters used during the matching.
filters = [f'JWST/NIRCam.{band}' for band in ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 
                                         'F410M', 'F444W']] + ['JWST/MIRI.F770W']
pivots = np.array([11542.61, 15007.44, 19886.48, 27617.40, 35683.62,
                    40822.38, 44043.15, 76393.34])
pivots /= 10000

# Set up the plot.
fig, ax = plt.subplots(2, 1, figsize=(3.78, 5.5), sharex=True)
fig.subplots_adjust(hspace=0.03)

# Extract the normalised and absolute SEDs from each region
all_seds = []
all_normed = []
for reg in regions:

    cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{reg}_{snapshot}_{sps_model}.hdf5'
    with h5py.File(cat_path, 'r') as cat:
        if reg == '38':
            continue

        # Store the absolute and normalised SEDs for each galaxy.
        phot_path = 'Galaxies/Stars/Photometry/Fluxes/stellar_total'
        seds = np.array([cat[f'{phot_path}/{filter}'][:] for filter in filters])

        norm_seds = normalise_seds(seds.T, scaler='Sum')
        all_seds.append(seds)
        all_normed.append(norm_seds.T)

        # Highlight the matches.
        idxs = cat['Galaxies/MasterRegionIndex'][:]
        for idx in idxs:

            if f'{reg}:{idx}' in identifiers:
                id = f'{reg}:{idx}'
                i = identifiers.index(id)

                s = idxs == idx

                if id in best_matches:
                    ax[0].plot(pivots, norm_seds[s].T, color=colours[i], linewidth=2, alpha=0.8, 
                               label=f'FRA-{i+1}', zorder=0.5)
                    ax[1].plot(pivots, np.log10(seds.T[s].T), color=colours[i], linewidth=2, 
                               alpha=0.8, label=f'FRA-{i+1}', zorder=0.5)
                else:
                    ax[0].plot(pivots, norm_seds[s].T, color='red', linewidth=1, alpha=0.7, 
                               zorder=0.25, linestyle='--')
                    ax[1].plot(pivots, np.log10(seds.T[s].T), color='red', linewidth=1, alpha=0.7, 
                               zorder=0.25, linestyle='--')

# # If there are any 'other' matches, add a legend entry for them.
if len(others) > 0:
    ax[0].plot([-1e6, -1e5], [1,2], color='red', linewidth=1, alpha=0.7, zorder=0.25, 
               linestyle='--', label='FRA-W')

# Plot the median FLARES SED in both spaces.
combined_seds = np.concatenate(all_seds, axis=1)
p16, p50, p84 = np.percentile(combined_seds, [16, 50, 84], axis=1)
ax[1].plot(pivots, np.log10(p50), color='grey', zorder=0.3, alpha=0.8, linewidth=1)
ax[1].fill_between(pivots, np.log10(p16), np.log10(p84), alpha=0.2, color='gray', zorder=0.3)

combined_norm = np.concatenate(all_normed, axis=1)
p16, p50, p84 = np.percentile(combined_norm, [16, 50, 84], axis=1)
ax[0].plot(pivots, p50, color='grey', label='FLARES', zorder=0.3, alpha=0.8)
ax[0].fill_between(pivots, p16, p84, alpha=0.2, color='gray', zorder=0.3)

# Plot the RQG SED in both spaces.
rqg_fluxes = np.array([45.2, 75.7, 108.3, 186.0, 469.6, 522.5, 527.6, 673.7])
rqg_errors = np.array([7.9, 6.7, 5.6, 9.3, 23.5, 26.1, 26.4, 94.8])

norm_rqg = rqg_fluxes/np.sum(rqg_fluxes)
norm_errors = rqg_errors/np.sum(rqg_fluxes)

ax[0].plot(pivots, norm_rqg, color='black', label = 'RQG', linewidth=2, alpha=0.8, zorder=0.4)
ax[0].fill_between(pivots, norm_rqg - norm_errors, norm_rqg + norm_errors, color='black', 
                   alpha=0.3, zorder=0.4)
ax[1].plot(pivots, np.log10(rqg_fluxes), color='black', label = 'RGQ', linewidth=2, 
           alpha=0.8, zorder=0.4)
ax[1].fill_between(pivots, np.log10(rqg_fluxes - rqg_errors), np.log10(rqg_fluxes + rqg_errors), 
                   color='black', alpha=0.3, zorder=0.4)

# Tidy up.
handles, labels = ax[0].get_legend_handles_labels()
legend_order = ['FRA-1', 'FRA-2', 'FRA-3', 'FRA-W', 'FLARES', 'RQG']
ordered_handles = []
ordered_labels = []
for name in legend_order:
    if name in labels:
        idx = labels.index(name)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

legend = ax[0].legend(ordered_handles, ordered_labels, frameon=True, ncols=1, framealpha=1, 
                      facecolor='white', fancybox=False, edgecolor='grey', fontsize=8, 
                      title='FSPS', loc='upper left')

ax[0].set_ylabel('$\\mathrm{F}_{\\nu} \ / \ \\sum \\mathrm{F}_{\\nu}$')
ax[1].set_ylabel('$\\log_{10}(\\mathrm{F}_{\\nu} \ / \ \\mathrm{nJy})$')
ax[1].set_xlabel('$\\lambda_{\\mathrm{obs}} \ / \ \\mu\\mathrm{m}$')

ax[0].set_ylim(0.01, 0.32)
ax[1].set_ylim(0.8, 3.2)
ax[1].set_yticks([1, 1.5, 2, 2.5, 3])

ax[1].set_xlim(1.15, 7.6)

secax = ax[0].secondary_xaxis('top')
secax.set_xlabel('$\\lambda_{\\mathrm{rest}} \ / \ \\mu\\mathrm{m}$', labelpad=6)
ticks = ax[0].get_xticks()
new_labels = [f"${t/(1+7.29):.2f}$" if t > 0 else "" for t in ticks]
secax.set_xticks(ticks)
secax.set_xticklabels(new_labels)

fig.savefig('plots/matched_fsps_seds.pdf', bbox_inches='tight', dpi=300)
plt.show()