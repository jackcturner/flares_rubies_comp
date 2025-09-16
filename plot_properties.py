import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
from utils import measure_t50

# Names and positions of each parameter to plot.
params = ['M', 'Z', 't', 'Av', 'SFR100', 'SFR50', 'SFR10']
labels = ['$\\mathrm{M}_{\\ast}$', '$\\mathrm{Z}_{\\ast}$', '$t_{50}$', 
          '$\\mathrm{A_{\\mathrm{V}}}$', '$\\mathrm{SFR}_{100}$', '$\\mathrm{SFR}_{50}$', 
          '$\\mathrm{SFR}_{10}$']
x_positions = np.arange(len(params))

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
snapshot = '008_z007p000'
identifiers = ['04:4', '10:125']
colours = ['green', 'orange']
markers = ['o', 's']
sps_models = ['fsps', 'bpass', 'bpass_alpha']

# Set up the plot.
fig, ax = plt.subplots(len(sps_models), 1, figsize=(3.78, 2*len(sps_models)), sharex=True)
fig.subplots_adjust(hspace=0.03)
ylim = 1.49

# To avoid overlapping points, apply a small shift to each identifier.
max_shift = 0.2
if len(identifiers) == 1:
    shifts = [0.0]
else:
    shifts = np.linspace(-max_shift, max_shift, len(identifiers))

# For each SPS model.   
with h5py.File(master_path, 'r') as master:
    for m, sps_model in enumerate(sps_models):

        # For each FLARES match.
        for i, id in enumerate(identifiers):
            reg, idx = id.split(':')
            idx = int(idx)
            key = f'{reg}/{snapshot}'

            # Extract the true properties.
            props_true = {}            
            props_true['M'] = master[f'{key}/Galaxy/Mstar_aperture/30'][idx] * 1e10
            props_true['Z'] = master[f'{key}/Galaxy/Metallicity/MassWeightedStellarZ'][idx] / 0.019
            props_true['SFR10'] = master[f'{key}/Galaxy/SFR_aperture/30/10Myr'][idx]
            props_true['SFR50'] = master[f'{key}/Galaxy/SFR_aperture/30/50Myr'][idx]
            props_true['SFR100'] = master[f'{key}/Galaxy/SFR_aperture/30/100Myr'][idx]
            props_true['t'] = measure_t50(master_path, reg, idx, snapshot) / 1e3

            # Get Av from the photometry.
            cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{reg}_{snapshot}_{sps_model}.hdf5'
            with h5py.File(cat_path, 'r') as cat:
                s = cat['Galaxies/MasterRegionIndex'][:] == idx
                phot_path = 'Galaxies/Stars/Photometry/Luminosities'
                props_true['Av'] = -2.5 * np.log10(cat[f'{phot_path}/stellar_total/V'][s] /
                                                   cat[f'{phot_path}/stellar_reprocessed/V'][s])

            # Load Prospector results.
            result_file = f'{base_path}/data/prospector_outputs/flares_prospector_{reg}_{idx}_{sps_model}_fiducial_dict.npy'
            results = np.load(result_file, allow_pickle=True).item()

            # Get the median and 1-sigma percentiles.
            props_pp = {}
            props_pp['Z'] = 10**np.percentile(results['logZ'], [16, 50, 84])
            props_pp['SFR10'] = np.percentile(results['sfr10'], [16, 50, 84])
            props_pp['SFR50'] = np.percentile(results['sfr50'], [16, 50, 84])
            props_pp['SFR100'] = np.percentile(results['sfr100'], [16, 50, 84])
            props_pp['t'] = np.percentile(results['t50'], [16, 50, 84])
            props_pp['M'] = 10**np.percentile(results['logM'], [16, 50, 84])
            props_pp['Av'] = np.percentile(results['dust2'], [16, 50, 84])

            # For each parameter.
            for j, param in enumerate(params):

                # Calculate the shift in dex.
                shift = np.log10(props_pp[param][1] / props_true[param])
                err_low = np.log10(props_pp[param][0] / props_pp[param][1])
                err_high = np.log10(props_pp[param][2] / props_pp[param][1])
                xpos = x_positions[j] + shifts[i]

                if j == 0:
                    label = f'FRA-{i+1}'
                else:
                    label = None
                
                # Plot it.
                # Use an arrow if outside the y-limits.
                if shift > ylim:
                    ax[m].annotate('', xy=(xpos, ylim), xytext=(xpos, ylim-0.25),
                                  arrowprops=dict(arrowstyle='-|>', color=colours[i], lw=1.5, fill=True))
                elif shift < -ylim:
                    ax[m].annotate('', xy=(xpos, -ylim), xytext=(xpos, -ylim+0.25),
                                  arrowprops=dict(arrowstyle='-|>', color=colours[i], lw=1.5, fill=True))
                else:
                    ax[m].scatter(xpos, shift, marker=markers[i], color=colours[i], s=30, 
                                  zorder=1.9, label=label)
                    ax[m].errorbar(xpos, shift, yerr=np.array([[abs(err_low)], [err_high]]), 
                                   fmt='none', capsize=2, zorder=1.8, color=colours[i], linewidth=1)
                    
        # Tidy up.
        ax[m].axhline(0, color='grey', linestyle='--', linewidth=0.8, zorder=0)

        if m == 0:
            ax[m].legend(loc='upper left', ncol=1, frameon=True, framealpha=1, facecolor='white',
                    fancybox=False, edgecolor='grey', fontsize=8)
            
        sps_model_ = sps_model.upper().replace('_', '\u2013').replace('ALPHA', r'\alpha')
        ax[m].set_ylabel(f'$\\log_{{10}}(P_{{\\mathrm{{{sps_model_}}}}} / P_{{\\mathrm{{true}}}})$')
        ax[m].set_xticks(x_positions, labels)
        ax[m].set_xlim(-0.5, len(params)-0.5)
        ax[m].set_ylim(-ylim, ylim)

fig.savefig('plots/properties_comparison_all.pdf', dpi=300, bbox_inches='tight')
plt.show()
