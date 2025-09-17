import re
import h5py
import numpy as np
from astropy.cosmology import Planck13 as cosmo

import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
plt.rcParams['axes.ymargin'] = 0.1

import cmasher as cmr
import matplotlib.colors as mcolors
cmap = plt.get_cmap("cmr.guppy_r")

# Colour top ticks by redshift to match gas history plot.
z_ticks = np.array(list(range(14, 6, -1)))
norm = mcolors.BoundaryNorm(boundaries=np.arange(min(z_ticks)-0.5, max(z_ticks)+0.6, 1), 
                            ncolors=cmap.N)

# FLARES relevant info.
master_path = 'flares.hdf5'
bh_details = 'data/blackhole_details.h5'
bh_mergers = 'data/bhmergers.h5'
snap = "008_z007p000"
match = re.search(r'_z(\d{3})p(\d{3})', snap)
redshift = float(f"{match.group(1)}.{match.group(2)}")
regions = (['00','01','02','03','04','05','06','07','08','09'] + 
           [str(i) for i in np.arange(10, 40)])

# Interested in these galaxies.
identifiers = ['04:4', '10:125']
colours=['green', 'orange']
bh_colours = ['blue', 'red']

# Set up the plot.
fig, ax = plt.subplots(len(identifiers), 1, figsize=(3.78, 2.5*(len(identifiers))), sharex=True)
fig.subplots_adjust(hspace=0.03)

# Bins of stellar age in Myr.
width = 20
bins = np.arange(0, cosmo.age(redshift).to('yr').value, step=width * 1e6)
bin_centres = (bins[1:] + bins[:-1]) / 2

x = np.repeat(bins, 2)[1:-1]
dt = np.diff(bins)

# Open the master and BH details files.
with (h5py.File(bh_details, 'r') as details, h5py.File(bh_mergers, 'r') as mergers, 
      h5py.File(master_path) as master):

    # For each analogue.
    for i, id in enumerate(identifiers):

        reg, idx = id.split(':')
        idx = int(idx)
        key = f"{reg}/{snap}/"

        # Get the star particle masses and ages.
        slength = master[f"{key}/Galaxy/S_Length"][:]
        sstart = np.insert(np.cumsum(slength), 0, 0)

        smass = master[f"{key}/Particle/S_MassInitial"][sstart[idx] : sstart[idx + 1]] * 1e10
        sage = master[f"{key}/Particle/S_Age"][sstart[idx] : sstart[idx + 1]] * 1e9

        # Calculate the binned SFR.
        H, _ = np.histogram(sage, bins=bins, weights=smass)

        sfr = H / dt
        y = np.repeat(sfr, 2)

        line1, = ax[i].plot((x / 1e6), y, color=colours[i], alpha=0.8, label=f'SFH', zorder=0.25)

        # Extract the IDs of the BHs.
        bh_length = master[f"{key}/Galaxy/BH_Length"][:]
        bh_start = np.insert(np.cumsum(bh_length), 0, 0)
        bh_ids = master[f"{key}/Particle/BH_ID"][bh_start[idx]:bh_start[idx+1]].astype(np.uint64)

        # Use bins of Universe age for this.
        uni_bins = cosmo.age(redshift).to('Myr').value - (bins / 1e6)         
        uni_centres = (uni_bins[1:] + uni_bins[:-1]) / 2

        rates = np.zeros(len(uni_centres))
        for bh_id in bh_ids:

            # Something wrong with the IDs so find the closest available 
            # in details file.
            bh_detail_ids = np.array(list(details[reg].keys())).astype(np.int64)
            index = np.argmin(np.abs(bh_detail_ids-bh_id))
            bh_id = bh_detail_ids[index]
            bh = details[f'{reg}/{bh_id}']

            # Average accretion rate within each bin.
            subgrid_mass = bh['BH_Subgrid_Mass'][()] * 1e10
            z = bh['z'][()]
            time = cosmo.age(z)
            _time = time.to('Myr').value

            average_accretion_rate = []
            for centre in uni_centres:
                starting_time = centre - (width/2)
                ending_time = centre + (width/2)
                ending_mass = np.interp(ending_time, _time, subgrid_mass)
                starting_mass = np.interp(starting_time, _time, subgrid_mass)

                average_accretion_rate.append((ending_mass - starting_mass)/width)
            average_accretion_rate = np.array(average_accretion_rate) /1e6
            rates += average_accretion_rate

        # Plot this on the right hand axis.
        ax2 = ax[i].twinx()
        norm_rate = rates
        norm_rate = np.repeat(norm_rate, 2)

        line2, = ax2.plot(x / 1e6, norm_rate, label=f'Accretion Rate', 
                          color=bh_colours[i], zorder=0.1)

        # Tidy up.
        ax[i].set_xlim(0.01, 650)
        ax[i].set_ylim(0)
        ax2.set_ylim(0)
  
        if i == 1:
            ax[i].set_xlabel("$t_{\\mathrm{LB}} \ / \ \\mathrm{Myr}$")
        ax2.set_ylabel('$\\dot{\\mathrm{M}}_{\\bullet} \ / \ \\mathrm{M_{\\odot}yr^{-1}}$')
        ax[i].set_ylabel("$\\mathrm{SFR} \ / \ \\mathrm{M_{\\odot}yr^{-1}}$")

        ax2.grid(visible=False)

        ax[i].tick_params(direction='in', top=False)
        ax[i].tick_params(which='minor', top=False)

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax[i].legend(lines, labels, frameon=True, framealpha=1, ncols=1, facecolor='white', 
                     fancybox=False, edgecolor='grey', fontsize=8, title=f'FRA-{i+1}', 
                     loc='upper left')

ax[1].invert_xaxis()

top_ax = ax[0].secondary_xaxis('top')
top_ax.set_xlabel("Redshift", labelpad=7)
labels = [20, 15, 12, 10, 9, 8, 7.29]
positions = [(cosmo.age(7)-cosmo.age(label)).value*1e3 for label in labels]
top_ax.set_xticks(positions)  
top_ax.set_xticklabels(labels)
top_ax.tick_params(which='minor', top=False)

# Create invisible overlay for top ticks.
overlay_ax = ax[0].twinx() 
overlay_ax.set_ylim(ax[0].get_ylim())
overlay_ax.axis("off")
overlay_ax.scatter(positions, [ax[0].get_ylim()[1]]*len(positions), c=labels, cmap=cmap, s=40,
                   marker='o', clip_on=False, zorder=10, edgecolor='k', linewidth=0.5)

fig.savefig('plots/sfh_accretion.pdf', dpi=300, bbox_inches='tight')
plt.show()


