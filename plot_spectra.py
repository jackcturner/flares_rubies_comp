import h5py
import numpy as np
from astropy.io import fits
import spectres

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

base_path = './'
snapshot = '008_z007p290'
sps_model = 'bpass'

# Load in the RQG spectrum.
rqg_spectrum = fits.open(f'{base_path}/data/rubies-uds31-v3_prism-clear_4233_166283.spec.fits')

rqg_grid = rqg_spectrum[1].data['wave']
rqg_fnu = rqg_spectrum[1].data['flux']
rqg_err = rqg_spectrum[1].data['err']

# Get the bin edges for plotting as a step function.
bin_centers = rqg_grid
bin_widths = np.diff(bin_centers)
bin_edges = np.concatenate([
    [bin_centers[0] - bin_widths[0]/2],
    bin_centers[:-1] + bin_widths/2,
    [bin_centers[-1] + bin_widths[-1]/2]
])

# Zoom on this region of the spectrum.
valid = (bin_edges > 3) & (bin_edges < 4.4)

# Get the matches.
identifiers = ['04:4', '10:125']
colours = ['green', 'orange']

# Set up the plot.
fig, ax = plt.subplots(1, 1, figsize=(8, 3.15))

# Plot the spectra of the best matches.
for i, id in enumerate(identifiers):

    reg, idx = id.split(':')
    idx = int(idx)

    cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{reg}_{snapshot}_{sps_model}.hdf5'
    with h5py.File(cat_path, 'r') as cat:

        s = cat['Galaxies/MasterRegionIndex'][:] == idx

        # Observer frame wavelength grid in microns.
        grid = (cat['Galaxies/Stars/Spectra/wavelength'][:] * (1+7.29))/10000 

        # Resample spectrum to NIRSpec resolution.
        spectrum = cat[f'Galaxies/Stars/Spectra/SpectralFluxDensities/stellar_total'][s][0]
        flares_spectres = spectres.spectres(rqg_grid, grid, spectrum)
        norm = np.nansum(flares_spectres)

        ax.step(bin_edges, (np.append(flares_spectres, flares_spectres[-1])/norm), where='post',
                color=colours[i], zorder=1-((0.25/len(identifiers))*(i+1)), 
                alpha=0.75, label=f'FRA-{i+1}')
# Plot the RQG spectrum.
norm = np.nansum(rqg_fnu)

upper = (rqg_fnu + rqg_err)/norm
lower = (rqg_fnu - rqg_err)/norm
upper = np.append(upper, upper[-1])
lower = np.append(lower, lower[-1])

ax.step(bin_edges, np.append(rqg_fnu, rqg_fnu[-1])/norm, where='post', color='black', alpha=0.8, 
        zorder=1.5, label='RQG')
ax.fill_between(bin_edges, lower, upper, step='post', color='black', alpha=0.3, zorder=1.5)

box = dict(facecolor='white', alpha=1, edgecolor='white')

# Add some emission/absorption line labels.
ax.axvline(0.382 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.380 * (1+7.29), 0.0012, '$\\mathrm{H}\\eta$', fontsize=8, bbox=box)
ax.axvline(0.388 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.386 * (1+7.29), 0.005, '$\\mathrm{H}\\zeta$', fontsize=8, bbox=box)
ax.axvline(0.392 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.390 * (1+7.29), 0.0011, '$\\mathrm{Ca} \ \\mathrm{K}$', fontsize=8, bbox=box)
ax.axvline(0.396 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.394 * (1+7.29), 0.0048, '$\\mathrm{Ca} \ \\mathrm{H}$', fontsize=8, bbox=box)
ax.text(0.394 * (1+7.29), 0.0043, '$\\mathrm{H}\\epsilon$', fontsize=8, bbox=box)
ax.axvline(0.410 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.408 * (1+7.29), 0.0011, '$\\mathrm{H}\\delta$', fontsize=8, bbox=box)
ax.axvline(0.435 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.433 * (1+7.29), 0.0045, '$\\mathrm{H}\\gamma$', fontsize=8, bbox=box)
ax.axvline(0.486 * (1+7.29), linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(0.484 * (1+7.29), 0.0015, '$\\mathrm{H}\\beta$', fontsize=8, bbox=box)
ax.axvline(5008.240 * (1+7.29) / 10000, linestyle='--', color='grey', zorder=0, linewidth=1)
ax.axvline(4958.91 * (1+7.29) / 10000, linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(4950 * (1+7.29) / 10000, 0.0015, '$[\mathrm{OIII}]$', fontsize=8, bbox=box)
ax.axvline(3726.03 * (1+7.29) / 10000, linestyle='--', color='grey', zorder=0, linewidth=1)
ax.text(3700 * (1+7.29) / 10000, 0.0038, '$[\mathrm{OII}]$', fontsize=8, bbox=box)

# Tidy up.
ax.set_xlim(2.95, 4.39)
ax.set_ylim(0.0008,0.0072)
ax.set_ylabel('$\\mathrm{F}_{\\nu} \ / \ \\sum \\mathrm{F}_{\\nu}$')
ax.set_xlabel('$\\lambda_{\\mathrm{obs}} \ / \ \\mu\\mathrm{m}$')
ax.tick_params(labelsize=8)

secax = ax.secondary_xaxis('top')
secax.set_xlabel('$\\lambda_{\\mathrm{rest}} \ / \ \\mu\\mathrm{m}$', labelpad=6)
ticks = ax.get_xticks()
new_labels = [f"${t/(1+7.29):.2f}$" if t > 0 else "" for t in ticks]
secax.set_xticks(ticks)
secax.set_xticklabels(new_labels)

legend = ax.legend(loc=2, frameon=True, ncols=1, framealpha=1, facecolor='white', 
                            fancybox=False, edgecolor='grey', fontsize=8)

plt.savefig('plots/norm_spectra.pdf', bbox_inches='tight', dpi=300)
plt.show()