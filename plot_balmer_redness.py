import h5py
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import cmasher
cmap = plt.get_cmap('cmr.guppy')

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
regions = (['00','01','02','03','04','05','06','07', '08', '09'] + 
           [str(i) for i in np.arange(10, 40)])
snapshot = '008_z007p290'
sps_model = 'bpass'

# Get the matches.
best_matches = ['04:4', '10:125']
colours = ['green', 'orange']
markers = ['o', 's']
others = ['00:361', '08:503', '04:14', '15:432', '06:755']
identifiers = best_matches + others

# Set up the plot.
fig, ax = plt.subplots(2, 1, figsize=(3.78, 5), sharex=True)
fig.subplots_adjust(hspace=0.03)

with h5py.File(master_path) as master:

    # For each region.
    for reg in regions:

        cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{reg}_{snapshot}_{sps_model}.hdf5'
        with h5py.File(cat_path, 'r') as cat:
            if reg == '38':
                continue

            # Extract the spectra.
            spectra = cat['Galaxies/Stars/Spectra/SpectralFluxDensities/stellar_total'][:]
            wavelengths = cat['Galaxies/Stars/Spectra/wavelength'][:]

            # Quantify the Balmer break using Wang+24 definition.
            win1 = (wavelengths >= 3620) & (wavelengths <= 3720)
            win2 = (wavelengths >= 4000) & (wavelengths <= 4100)
            balmer_break = np.median(spectra[:, win2], axis=1) / np.median(spectra[:, win1], axis=1)
            
            # Quantitfy the 'redness' of the SED
            phot_path = 'Galaxies/Stars/Photometry/Fluxes/stellar_total'
            col1 = -2.5 * np.log10(cat[f'{phot_path}/JWST/NIRCam.F150W'][:] /
                                cat[f'{phot_path}/JWST/NIRCam.F444W'][:])

            # and magnitude in F444W.
            m444w = -2.5 * np.log10(cat[f'{phot_path}/JWST/NIRCam.F444W'][:] / 1e9) + 8.90

            # Calculate the sSFR from the master file.
            idxs = cat['Galaxies/MasterRegionIndex'][:]
            key = f'{reg}/008_z007p000/Galaxy'

            ssfrs = []
            for idx, col1_, m444w_, balmer_break_ in zip(idxs, col1, m444w, balmer_break):
                mass = master[f'{key}/Mstar_aperture/30'][idx] * 1e10
                ssfrs.append(np.log10(1e9 * master[f'{key}/SFR_aperture/30/50Myr'][idx] / mass))

                # Highlight the matches.
                if f'{reg}:{idx}' in identifiers:
                    id = f'{reg}:{idx}'
                    i = identifiers.index(id)

                    if id in best_matches:
                        ax[0].scatter(col1_, balmer_break_, s=40, label=f'FRA-{i+1}', 
                                      color=colours[i], marker=markers[i], zorder=1.5)
                        ax[1].scatter(col1_, m444w_, s=40, label=f'FRA-{i+1}', color=colours[i], 
                                      marker=markers[i], zorder=1.5)
                    else:
                        ax[0].scatter(col1_, balmer_break_, s=20, color='red', marker='x', 
                                      zorder=1.4)
                        ax[1].scatter(col1_, m444w_, s=20, color='red', marker='x', zorder=1.4)


            # Plot break stength and magnitude vs redness.
            ax[0].scatter(col1, balmer_break, c=ssfrs, s = 0.1, vmin=-0.7, vmax=1.2,
                          cmap=cmap, zorder=1)
            sc = ax[1].scatter(col1, m444w, c=ssfrs, s = 0.1, vmin=-0.7, vmax=1.2,
                               cmap=cmap, zorder=1)

# If there are any 'other' matches, add a legend entry for them.
if len(others) > 0:
    ax[0].scatter(-99, -99, s=20, color='red', label='FRA-W', marker='x')

# Calculate redness of RQG.
rqg_fluxes = np.array([45.2, 75.7, 108.3, 186.0, 469.6, 522.5, 527.6, 673.7])/1e9
rqg_errors = np.array([7.9, 6.7, 5.6, 9.3, 23.5, 26.1, 26.4, 94.8])/1e9
rqg = np.column_stack((rqg_fluxes, rqg_errors))

# Calculate magnitudes and get upper and lower limits.
rqg_mag = -2.5 * np.log10(rqg_fluxes) + 8.90
magplus = (-2.5 * np.log10(rqg_fluxes - rqg_errors) + 8.90) - rqg_mag
magminus = rqg_mag - (-2.5 * np.log10(rqg_fluxes + rqg_errors) + 8.90)

# Get the redness with error.
col1 = -2.5*np.log10(rqg[1, 0]/rqg[5, 0])
col1_err = np.array([[np.sqrt((magminus[1]**2)+(magplus[5]**2))],
                     [np.sqrt((magplus[1]**2)+(magminus[5]**2))]])

# Add RQG to the plot.
ax[0].errorbar(col1, 2.53, yerr=np.array([[0.21],[0.21]]), xerr=col1_err, c='black', fmt='none', 
               linewidth=1, capsize=2, zorder=1.6)
ax[0].scatter(col1, 2.53, color='black', label='RQG', s=50, marker='*', zorder=1.6)

ax[1].errorbar([col1], [rqg_mag[5]], xerr=col1_err, c='black', fmt='none', linewidth=1, 
               capsize=2, zorder=1.6)
ax[1].scatter(col1, rqg_mag[5], color='black', label='RQG', s=50, marker='*', zorder=1.6)

#  Tidy up.
ax[0].legend(frameon=True, framealpha=1, ncols=1, facecolor='white', 
             fancybox=False, edgecolor='grey', fontsize=8)

ax[0].set_ylabel('$\\mathrm{F}_{4000-4100} \ / \ \\mathrm{F}_{3620-3720}$')
ax[1].set_xlabel('$\\mathrm{F150W - F444W}$')
ax[1].set_ylabel('$\\mathrm{m}_{\\mathrm{F444W}}$')

ax[0].set_xlim(-0.1, 2.4)
ax[0].set_ylim(0.6, 3.1)
ax[1].set_ylim(22.1, 29.9)
ax[1].invert_yaxis()

pos = ax[0].get_position()
cbar_height = 0.02
cbar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, cbar_height])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label("$\\log_{10}(\\mathrm{sSFR}_{50} \\ \ / \\ \\mathrm{Gyr}^{-1})$", labelpad=7)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(size=2)
cb.ax.minorticks_off()

fig.savefig('plots/balmer_redness.pdf', bbox_inches='tight', dpi=300)
plt.show()
