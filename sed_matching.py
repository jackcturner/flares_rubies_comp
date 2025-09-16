import numpy as np
from astropy.table import Table
from utils import read_sed_from_hdf5, resample_fluxes, normalise_seds, find_matches_chi2

# Use the shifted snapshot.
base_path = './'
snapshot = '008_z007p290'
regions = (['00','01','02','03','04','05','06','07','08','09'] + 
           [str(i) for i in np.arange(10, 40)])
phot_path = 'Galaxies/Stars/Photometry/Fluxes/stellar_total'
sps_model = 'fsps'

# Filters to use.
filters = [f'JWST/NIRCam.{band}' for band in ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 
                                         'F410M', 'F444W']] + ['JWST/MIRI.F770W']

pivots = np.array([11542.61, 15007.44, 19886.48, 27617.40, 35683.62,
                    40822.38, 44043.15, 76393.34])
pivots /= 10000

# The RQG photometry in nJy.
rqg_fluxes = [45.2, 75.7, 108.3, 186.0, 469.6, 522.5, 527.6, 673.7]
rqg_errors = [7.9, 6.7, 5.6, 9.3, 23.5, 26.1, 26.4, 94.8]
rqg = np.column_stack((rqg_fluxes, rqg_errors))

# Various scalers supported, just using 'Sum' for now.
scalers = ['Sum']
#scalers = ['Sum', 'Standard', 'MinMax', 0, 1, 2, 3, 4, 5, 6, 7]

# The number of resampling iterations.
its = 10000

# For each region.
flares_seds = 0
region_num = []
indices = []
for region in regions:
    if region == '38':
        continue

    # Read SEDs and master file indices.
    cat_path = f'{base_path}/data/{snapshot}/RUBIES_COMP_{region}_{snapshot}_{sps_model}.hdf5'
    seds, indices_ = read_sed_from_hdf5(cat_path, filters, phot_path=phot_path)

    if isinstance(flares_seds, int):
        flares_seds = seds
    else:
        flares_seds = np.vstack((flares_seds, seds))

    # Store region numbers and indices.
    region_num += [region]*len(indices_)
    indices += [int(i) for i in indices_]

# Store the matching information of each source.
region_num = np.array(region_num)
indices = np.array(indices)

match_table = Table()
match_table['region'] = region_num
match_table['index'] = indices

# For each scaler.
for scaler in scalers:

    print(f'Scaler: {scaler}')

    # Get an array of resampled and normalised RQG SEDS.
    rqg_seds = resample_fluxes(rqg, its)
    rqg_seds = normalise_seds(rqg_seds, scaler=scaler)

    # Normalise the FLARES galaxies in the same way.
    norm_flares = normalise_seds(flares_seds, scaler=scaler)

    # Match each iteration.
    match_info = np.zeros(shape=(norm_flares.shape[0], 2))
    for sed in rqg_seds:

        # Find the best matching galaxies by shape.
        best, distances = find_matches_chi2(sed, norm_flares)

        # Record best match and distances.
        match_info[best, 1] += 1
        match_info[:, 0] += distances

    if isinstance(scaler, str):
        label = scaler
    else:
        label = filters[scaler]

    match_table[f'{label}_Nbest'] = match_info[:, 1]
    match_table[f'{label}_distance'] = match_info[:, 0]

match_table.write(f'{base_path}/data/flares_{sps_model}_rqg_matching.fits', overwrite=True)