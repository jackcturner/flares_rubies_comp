import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

# FLARES relevant info.
base_path = './'
master_path = 'flares.hdf5'
snapshot = '008_z007p000'

# The SPS models to compare.
sps_models = ['fsps', 'fsps_intrinsic', 'bpass', 'bpass_alpha']

# Get the matches.
identifiers = ['04:4', '10:125']
colors = ['green', 'orange']
markers = ['o', 's']

# Show fiducial and high-Z runs with different alphas.
suffixes = ['_fiducial', '_highZ']
alphas = [1.0, 0]
result_dict = {suffix: {id: [] for id in identifiers} for suffix in suffixes}
err_dict = {suffix: {id: [] for id in identifiers} for suffix in suffixes}

with h5py.File(master_path, 'r') as master:

    for i, identifier in enumerate(identifiers):
        reg, idx = identifier.split(':')
        idx = int(idx)

        # Get the true metallicity.
        key = f'{reg}/{snapshot}/Galaxy'
        true_Z = np.log10(master[f'{key}/Metallicity/MassWeightedStellarZ'][idx] / 0.019)

        # Get values inferred by each fit.
        for sps_model in sps_models:
            for suffix in suffixes:
                file_path = f'{base_path}/data/prospector_outputs/flares_prospector_{reg}_{idx}_{sps_model}{suffix}_dict.npy'

                try:
                    results = np.load(file_path, allow_pickle=True).item()
                    fit_logZ = np.percentile(results['logZ'], [16, 50, 84])
                    diff = fit_logZ - true_Z
                    result_dict[suffix][identifier].append(diff[1])
                    err_dict[suffix][identifier].append([diff[1] - diff[0], diff[2] - diff[1]])

                except FileNotFoundError:
                    result_dict[suffix][identifier].append(np.nan)
                    err_dict[suffix][identifier].append([np.nan, np.nan])

# Set up the plot.
fig, ax = plt.subplots(figsize=(3.78, 3.78))
x = np.arange(len(sps_models))
width = 0.35

for i, identifier in enumerate(identifiers):
    for suffix, alpha in zip(suffixes, alphas):

        diffs = result_dict[suffix][identifier]
        errs = np.array(err_dict[suffix][identifier]).T

        label = f'FRA-{i+1}' if (suffix == suffixes[0] and alpha == 1.0) else None
        ax.errorbar(x + i*width, diffs, yerr=errs, fmt='none', color=colors[i], capsize=4, alpha=alpha)
        ax.scatter(x + i*width, diffs, marker=markers[i], color=colors[i], alpha=alpha, label=label)

# Clean up model names for plotting.
sps_models = [model.upper().replace('_', '\u2013').replace('ALPHA', r'\alpha') for model in sps_models]

# True value.
ax.axhline(0, color='grey', linestyle='--', lw=1, zorder=0)

ax.set_xlabel('SPS Model')
ax.set_ylabel('$\\log_{10}(\\mathrm{Z_{fit}} \ / \ \\mathrm{Z_{true}})$')

ax.set_xticks(x + width/2)
ax.set_xticklabels([f'$\\mathrm{{{model}}}$' for model in sps_models])
ax.tick_params(axis='x', which='minor', bottom=False, top=False)

ax.legend(frameon=True, framealpha=1, ncols=1, facecolor='white', 
          fancybox=False, edgecolor='grey', fontsize=8)
ax.set_ylim(-0.79, 0.19)

plt.tight_layout()
plt.savefig('plots/metallicity_offset.pdf', bbox_inches='tight', dpi=300)
plt.show()




