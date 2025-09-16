import h5py
import numpy as np
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import cmasher as cmr
cmap2 = plt.get_cmap('cmr.lavender')
cmap = plt.get_cmap('cmr.cosmic')

from utils import measure_t50

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
snapshot = '008_z007p000'
sps_model = 'bpass'
regions = (['00','01','02','03','04','05','06','07','08','09'] + 
           [str(i) for i in np.arange(10, 40)])

# Get the matches.
best_matches = ['04:4', '10:125']
colours = ['green', 'orange']
markers = ['o', 's']
others = ['00:361', '08:503', '04:14', '15:432', '06:755']
identifiers = best_matches + others

# Set up the plot.
fig, ax = plt.subplots(2,1, figsize=(3.78, 6), sharex=True)
fig.subplots_adjust(hspace=0.03)

# Extract stellar mass, metallicity, sSFR and t50 from the master file.
masses = []
Zs = []
ssfrs = []
ages_50 = []
with h5py.File(master_path, 'r') as master:

    # For each region.
    for reg in regions:


        # Get the indices of galaxies in the photometry sample.
        cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{reg}_{snapshot}_{sps_model}.hdf5'
        with h5py.File(cat_path, 'r') as cat:
            if (reg == '38'):
                continue

            key = f'{reg}/008_z007p000/Galaxy'
            idxs = cat['Galaxies/MasterRegionIndex'][:]

            # Calculate each quantity.
            for idx in idxs:
                mass = np.log10(master[f'{key}/Mstar_aperture/30'][idx] * 1e10)
                Z = np.log10(master[f'{key}/Metallicity/MassWeightedStellarZ'][idx] / 0.019)
                ssfr = np.log10(1e9 * master[f'{key}/SFR_aperture/30/50Myr'][idx] / (10**mass))

                # Highlight the matches.
                if f'{reg}:{idx}' in identifiers:
                    id = f'{reg}:{idx}'
                    i = identifiers.index(id)

                    if id in best_matches:
                        ax[0].scatter(mass, ssfr, label=f'FRA-{i+1}', color=colours[i], 
                                      s=40, marker=markers[i])
                        ax[1].scatter(mass, Z, label=f'FRA-{i+1}', color=colours[i], 
                                      s=40, marker=markers[i])
                    else:
                        ax[0].scatter(mass, ssfr, color='red', s=20, marker='x')
                        ax[1].scatter(mass, Z, color='red', s=20, marker='x')  
                else:
                    masses.append(mass)
                    Zs.append(Z)
                    ssfrs.append(ssfr)
                    ages_50.append(measure_t50(master_path, reg, idx))

# If there are any 'other' matches, add a legend entry for them.
if len(others) > 0:
    ax[1].scatter(-99, -99, color='red', s=20, label='FRA-W', marker='x')

# Plot the FLARES results.
masses = np.array(masses)
ssfrs = np.array(ssfrs)
Zs = np.array(Zs)

sc_msfr = ax[0].scatter(masses, ssfrs, s=1, c=np.log10(ages_50), cmap=cmap2, zorder=0)
sc_mZ = ax[1].scatter(masses, Zs, s=1, c=np.log10(ages_50), cmap=cmap2, zorder=0)

# Add median lines.
x = np.linspace(masses.min()*0.99, masses.max()*1.01, 100)
bins = np.linspace(masses.min(), masses.max(), 20)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

median_Z, _, _ = binned_statistic(masses, Zs, statistic='median', bins=bins)
ax[1].plot(bin_centers, median_Z, linewidth=2.5, color='grey', alpha=0.8)
median_sfr, _, _ = binned_statistic(masses, ssfrs, statistic='median', bins=bins)
ax[0].plot(bin_centers, median_sfr, linewidth=2.5, color='grey', alpha=0.8)

# Helper function for Monte Carlo sSFR sampling from asymmetric errors.
def monte_carlo_ssfr(mass, mass_err_lo, mass_err_hi, sfr, sfr_err_lo, sfr_err_hi, N=10000):
    mass_samples = np.where(
        np.random.rand(N) < 0.5,
        np.random.normal(mass, mass_err_lo, N),
        np.random.normal(mass, mass_err_hi, N),
    )
    sfr_samples = np.where(
        np.random.rand(N) < 0.5,
        np.random.normal(sfr, sfr_err_lo, N),
        np.random.normal(sfr, sfr_err_hi, N),
    )
    valid = (mass_samples > 0) & (sfr_samples >= 0)
    ssfr_samples = sfr_samples[valid] / mass_samples[valid]
    ssfr_median = np.log10(np.median(ssfr_samples) * 1e9)
    ssfr_err_lo = ssfr_median - np.log10(np.percentile(ssfr_samples, 16) * 1e9)
    ssfr_err_hi = np.log10(np.percentile(ssfr_samples, 84) * 1e9) - ssfr_median
    return ssfr_median, ssfr_err_lo, ssfr_err_hi

# Calculate the sSFR for the fiducial run.
mass_fid = 10**10.23
mass_err_lo_fid = mass_fid - (10**10.19)
mass_err_hi_fid = (10**10.27) - mass_fid
sfr_fid = 0.83
sfr_err_lo_fid = 0.76
sfr_err_hi_fid = 11.11
ssfr_median_fid, ssfr_err_lo_fid, ssfr_err_hi_fid = monte_carlo_ssfr(
    mass_fid, mass_err_lo_fid, mass_err_hi_fid, sfr_fid, sfr_err_lo_fid, sfr_err_hi_fid)

ax[0].scatter(np.log10(mass_fid), ssfr_median_fid, s=50, color='black', 
              label='RQG (Fiducial)', marker='*')
ax[0].errorbar(np.log10(mass_fid), ssfr_median_fid, xerr=[[0.04],[0.04]], 
               yerr=[[ssfr_err_lo_fid],[ssfr_err_hi_fid]], color='black', 
               fmt='none', linewidth=1, capsize=2)

# Calculate the sSFR for the highZ run.
mass_hz = 10**10.19
mass_err_lo_hz = mass_hz - (10**10.15)
mass_err_hi_hz = (10**10.23) - mass_hz
sfr_hz = 2.13
sfr_err_lo_hz = 1.92
sfr_err_hi_hz = 5.54
ssfr_median_hz, ssfr_err_lo_hz, ssfr_err_hi_hz = monte_carlo_ssfr(
    mass_hz, mass_err_lo_hz, mass_err_hi_hz, sfr_hz, sfr_err_lo_hz, sfr_err_hi_hz)

ax[0].scatter(np.log10(mass_hz), ssfr_median_hz, s=50, color='saddlebrown', 
              label='High-Z', marker='*')
ax[0].errorbar(np.log10(mass_hz), ssfr_median_hz, xerr=[[0.04],[0.04]], 
               yerr=[[ssfr_err_lo_hz],[ssfr_err_hi_hz]], color='saddlebrown', 
               fmt='none', linewidth=1, capsize=2)

# Add quiescence selection region.
ax[0].axhspan(ymin=-2, ymax=np.log10(0.2 / cosmo.age(7.29).value), facecolor='grey', 
              alpha=0.3, zorder=0)
ax[0].axhline(np.log10(0.2 / cosmo.age(7.29).value), color = 'grey', linestyle='--', 
              alpha=1, zorder=0)

# Plot metallicity points.
ax[1].errorbar(10.23, -0.94, xerr=[[0.04],[0.04]], yerr=[[0.04],[0.05]], color='black', fmt='none',
               linewidth=1, capsize=2)
ax[1].scatter(10.23, -0.94, s=50, color='black', label='Fiducial', marker='*')
ax[1].errorbar(10.19, 0.07, xerr=[[0.04],[0.04]], yerr=[[0.11],[0.08]], color='saddlebrown', 
               fmt='none', linewidth=1, capsize=2)
ax[1].scatter(10.19, 0.07, s=50, color='saddlebrown', label='High-Z', marker='*')

# Tidy up.
ax[0].set_ylabel("$\\log_{10}(\\mathrm{sSFR}_{50} \ / \ \\mathrm{Gyr}^{-1})$")
ax[1].set_ylabel('$\\log_{10}(\\mathrm{Z}_{\\ast} \ / \ \\mathrm{Z}_{\\odot})$')
ax[1].set_xlabel('$\\log_{10}(\\mathrm{M}_{\\ast} \ / \ \\mathrm{M}_{\\odot})$')

pos = ax[0].get_position()
cbar_height = 0.02
cbar_ax = fig.add_axes([pos.x0, pos.y1, pos.width, cbar_height])
cb = fig.colorbar(sc_msfr, cax=cbar_ax, orientation='horizontal')
cb.set_label("$\\log_{10}(\\mathrm{sSFR}_{50} \\ \ / \\ \\mathrm{Gyr}^{-1})$", labelpad=7)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xticks([1.5, 1.7, 1.9, 2.1, 2.3, 2.5])
cb.ax.tick_params(size=2)
cb.ax.minorticks_off()

ax[1].legend(frameon=True, framealpha=1, ncols=2, facecolor='white', 
          fancybox=False, edgecolor='grey', fontsize=8)
ax[0].set_ylim(-1.6, 1.6)
ax[1].set_xlim(8.5, 10.8)
ax[1].set_ylim(-1.6, 0.35)

fig.savefig('plots/mass_ssfr_z_relation.pdf', bbox_inches='tight', dpi=300)
plt.show()