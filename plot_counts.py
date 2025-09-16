import h5py
import numpy as np
import scipy.stats as st
from astropy.table import Table
from astropy.cosmology import Planck13 as cosmo

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

from itertools import product
from collections import defaultdict

# Function to get Poisson confidence intervals. Returns zero uncertainty
# when n = 0. Not strictly correct, but ensures we don't have huge 
# upper errors.
def poisson_confidence_interval(counts, p=0.68):    
    lower = []
    upper = []

    for n in counts:
    
        if n>0:   
            interval=(st.chi2.ppf((1.-p)/2.,2*n)/2.,st.chi2.ppf(p+(1.-p)/2.,2*(n+1))/2.)  
            lower.append(n - interval[0])
            upper.append(interval[1] - n)     
        
        else:                    
            lower.append(0)
            upper.append(0)
    
    intervals = np.column_stack([lower, upper])
    
    return intervals

# FLARES relevant info.
master_path = 'flares.hdf5'
snapshot = '008_z007p000'
vsim = (4/3) * np.pi * ((14/0.6777)**3)

# Region numbers and their weights.
regions = (['00','01','02','03','04','05','06','07','08', '09'] + 
           [str(i) for i in np.arange(10, 40)])
weight_data = Table.read('data/region_weights.fits')
weights = weight_data['weight']

# Identifiers of RQG like galaxies.
identifiers = ['04:4', '10:125']

# Systematic choices.
apertures = [1, 3, 5, 10, 20, 30, 40, 50]
timescales = ['inst', '5Myr', '10Myr', '20Myr', '50Myr', '100Myr']
mass_limits = np.arange(8, 11, step=0.25)
ticks = [8.5, 9.0, 9.5, 10.0]
q_limits = [np.log10(0.2 / cosmo.age(7).value), -1, -2]
combinations = list(product(apertures, timescales, q_limits))

# Set up the plot.
fig, ax = plt.subplots(1,1, figsize=(3.78, 4))

mean_densities = []
with h5py.File(master_path, 'r') as master:
    # Iterate over each combination
    for mass_limit in mass_limits:
        q_counts = np.zeros([len(combinations), len(regions)])

        for i, (aperture, timescale, q_limit) in enumerate(combinations):

            for reg in regions:
                key = f'{reg}/008_z007p000/Galaxy'
                mass = master[f'{key}/Mstar_aperture/{aperture}'][:] * 1e10
                ssfr = np.log10(1e9 * master[f'{key}/SFR_aperture/{aperture}/{timescale}'][:]/mass)

                s = (np.log10(mass) > mass_limit) & (ssfr < q_limit)
                q_counts[i, regions.index(reg)] += np.sum(s)

        # Get the mean count across combinations.
        q_mean = np.mean(q_counts, axis=0)
        C = np.sum(q_mean * weights / vsim)
        mean_densities.append(np.log10(C))

        # At set points, get the maximum density.
        if round(mass_limit, 2) in ticks:
            max_idx = np.argmax(np.sum(q_counts * weights, axis=1))

            q_max = q_counts[max_idx, :]
            C_max = np.sum(q_max * weights / vsim)

            # Get the upper and lower Poisson errors.
            p_err = poisson_confidence_interval(q_max)
            q_lower = (np.log10(C_max) - 
                       np.log10(C_max - np.sqrt(np.sum((p_err[:, 0] * weights / vsim)**2))))
            q_upper = (np.log10(C_max + np.sqrt(np.sum((p_err[:, 1] * weights / vsim)**2))) - 
                       np.log10(C_max))
            
            # Plot the maximum value at this mass limit.
            plt.scatter(mass_limit, np.log10(C_max), color='white', edgecolors='blue', marker='v', 
                        s=30, zorder=1.5, alpha=1)
            plt.errorbar(mass_limit, np.log10(C_max), yerr=[[q_lower], [q_upper]], color='blue', 
                         capsize=3, linewidth=1, fmt='none', zorder=1.5, alpha=1)

    # Calculate the RQG analogue number density.
    q_rub = 0
    q_rub_lo = 0
    q_rub_hi = 0
    mass_limit = 1e6

    # Group identifiers by region.
    region_to_indices = defaultdict(list)
    for id in identifiers:
        reg, idx = id.split(':')
        region_to_indices[reg].append(int(idx))

    # Iterate over each region with at least one identifier.
    for reg, idx_list in region_to_indices.items():
        n_sources = len(idx_list)
        if n_sources == 0:
            continue

        # Update the lower mass limit.
        masses = np.log10(master[f'{reg}/008_z007p000/Galaxy/Mstar_aperture/30'][:] * 1e10)
        region_masses = masses[idx_list]
        min_mass = np.min(region_masses)
        if min_mass < mass_limit:
            mass_limit = min_mass

        # Update the counts and uncertainties.
        weight = weights[regions.index(reg)]
        p_err_reg = poisson_confidence_interval([n_sources])[0]
        q_rub += (n_sources * weight / vsim)
        q_rub_lo += (p_err_reg[0] * weight / vsim)**2
        q_rub_hi += (p_err_reg[1] * weight / vsim)**2

# Plot the mean counts.
ax.plot(mass_limits, mean_densities, color='blue', label='Mean', zorder=0.9, 
        markersize=5, alpha=0.8)

# Add legend entry for the max.
ax.scatter(-99, -99, color='white', edgecolors='blue', marker='v', label='Max', s=30)

# Plot the RQG analogue point.
q_lower = np.log10(q_rub) - np.log10(q_rub - np.sqrt(np.sum(q_rub_lo)))
q_upper = np.log10(q_rub + np.sqrt(np.sum(q_rub_hi))) - np.log10(q_rub)

ax.scatter(mass_limit, np.log10(q_rub), color='purple', marker='D', 
           label='RQG Analogues', s=30, zorder=2)
ax.errorbar(mass_limit, np.log10(q_rub), yerr=[[q_lower], [q_upper]], color='purple', capsize=3, 
            linewidth=1, fmt='none', zorder=2, alpha=0.8)

# Add observed points.
ax.errorbar(9.4, -6.57, yerr=np.array([[0.76], [0.52]]), color='grey', 
            fmt='none', capsize=3, zorder=1)
ax.scatter(9.4, -6.57, color='white', s=30, zorder=1, marker='o', edgecolors='grey')
ax.errorbar(9.45, -6.09, yerr=np.array([[0.3], [0.34]]), color='grey', 
            fmt='none', capsize=3, zorder=1)
ax.scatter(9.45, -6.09, color='grey', s=30, zorder=1, marker='o', label = 'Merlin+25 $(7<z<10)$', 
           edgecolors='grey')

ax.scatter(10.05, np.log10(0.03e-5), color='grey', s=30, 
           label='Baker+25 $(6 < z < 7.5)$', zorder=1, marker='p')
lolim = np.log10(0.03e-5) - np.log10(0.03e-5 - 0.02999e-5)
uplim = np.log10(0.03e-5 + 0.05e-5) - np.log10(0.03e-5)
ax.errorbar(10.05, np.log10(0.03e-5), yerr=np.array([[lolim], [uplim]]), color='grey', 
            fmt='none', capsize=3, zorder=1)
ax.scatter(9.55, np.log10(0.07e-5), color='grey', s=30, zorder=1, marker='p')
lolim = np.log10(0.07e-5) - np.log10(0.07e-5 - 0.06e-5)
uplim = np.log10(0.07e-5 + 0.06e-5) - np.log10(0.07e-5)
ax.errorbar(9.55, np.log10(0.07e-5), yerr=np.array([[lolim], [uplim]]), color='grey', 
            fmt='none', capsize=3, zorder=1)

ax.scatter(9.95, -5.8, color='black', s=50, label='Weibel+25 $(z=7.29)$', 
           zorder=1, marker='*')
ax.errorbar(9.95, -5.8, yerr=np.array([[0.8], [0.5]]), color='black', fmt='none', 
            capsize=3, zorder=1)

# Tidy up.
ax.set_xlabel('$\\log_{10}(\\mathrm{M_{min}} \ / \ \\mathrm{M_{\\odot}})$')
ax.set_ylabel('$\\log_{10}(\\mathrm{N_{Q}(M_{\\ast} > M_{min})} \ / \ \\mathrm{Mpc^{-3}})$')
ax.set_xticks(ticks, [f'${limit}$' for limit in ticks])

ax.set_ylim(-8.9, -4.51)
ax.set_xlim(8.3, 10.2)

ax.legend(loc='lower left', frameon=True, framealpha=1, facecolor='white', fancybox=False, 
          edgecolor='grey', fontsize=8, ncols=1)

fig.savefig('plots/number_density.pdf', dpi=300, bbox_inches='tight')
plt.show()