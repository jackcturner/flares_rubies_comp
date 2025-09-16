import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr
import h5py
from scipy.interpolate import interp1d

def resample_fluxes(sed, n):
    """
    Resample fluxes from a Gaussian based on their uncertainties..
    
    Arguments
    ---------
    sed (numpy.ndarray)
        2D array describing an SED. First column stores fluxes the and
        the second errors.
    n (int)
        The number of resampling iterations.
        
    Returns
    -------
    resampled (numpy.ndarray)
        Array of resampled fluxes with shape (n, len(sed[:, 0]))    
    """

    # Resample from the Gaussian.
    resampled = np.random.normal(loc=sed[:, 0], scale=sed[:, 1], size=(n, len(sed[:, 0])))

    return resampled

def normalise_seds(seds, scaler='MinMax'):
    """
    Normalise a 2D array of SEDs with shape.
    
    Arguments
    ---------
    seds (numpy.ndarray)
        2D array of SEDs to normalise. Each row is assumed to be an SED.
    scaler (str/int)
        The type of scaler to use. If an integer is provided, normalise 
        by the flux at that SED index.
    
    Returns
    -------
    norm_seds (numpy.ndarray)
        2D array of normalised SEDs.
    """

    # If a 1D array is provided, reshape it to 2D.
    orig_shape = seds.shape
    if seds.ndim == 1:
        seds = seds.reshape(1, -1)

    # Scale using the requested approach.
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        norm_seds = scaler.fit_transform(seds.T).T

    elif scaler == 'Standard':
        scaler = StandardScaler()
        norm_seds = scaler.fit_transform(seds.T).T

    elif scaler == 'Sum':
        sed_sums = np.sum(seds, axis=1)
        norm_seds = seds / sed_sums[:, np.newaxis]

    # Normalise by the flux at the provided index.
    elif isinstance(scaler, int):
        norm_seds = seds / seds[:, scaler].reshape(-1, 1)
    
    else:
        raise KeyError(f'{scaler} is not a valid scaler.')
    
    # If original input was (N,), convert back.
    if len(orig_shape) == 1:
        return norm_seds.flatten() 
    
    return norm_seds  

def find_matches(reference, seds):
    """
    Use Pearson correlation to match SEDs to a reference.
    
    Arguments
    ---------
    reference (numpy.ndarray)
        1D array of normalised reference SED fluxes.
    seds (numpy.ndarray)
        2D array of normalised SEDs to match to the reference. 
        Assumes each row corresponds to an SED.

    Returns
    -------
    match_index (int)
        Index into distances of closest matching SED.
    distances (numpy.ndarray)
        1D array of distances based on the Pearson coefficient.
    """

    # For each SED.
    distances = []
    for sed in seds:

        # Calculate the Pearson correlation coefficient.
        corr_coeff, _ = pearsonr(reference, sed)

        # Distance is probability of not being correlated.
        dist = 1 - corr_coeff
        distances.append(dist)
    
    # Return the index of the best matching source and the distances.
    match_index = np.argmin(distances)
    distances = np.array(distances)

    return match_index, distances

def find_matches_chi2(reference, seds):
    """
    Use Chi2 to match SEDs to a reference.
    
    Arguments
    ---------
    reference (numpy.ndarray)
        1D array of normalised reference SED fluxes.
    seds (numpy.ndarray)
        2D array of normalised SEDs to match to the reference. 
        Assumes each row corresponds to an SED.

    Returns
    -------
    match_index (int)
        Index into chi2s of closest matching SED.
    chi2s (numpy.ndarray)
        1D array of Chi2s.
    """

    # For each SED.
    chi2s = []
    for sed in seds:

        # Calculate the Chi2.
        chi2 = ((reference - sed)**2)/reference
        chi2s.append(np.sum(chi2))
    
    # Return the index of the best matching source and the distances.
    match_index = np.argmin(chi2s)
    chi2s = np.array(chi2s)

    return match_index, chi2s

def read_sed_from_hdf5(path, filters, phot_path='', conversion=1.0):
    """
    Move SEDs stored in an hdf5 file to an array.
    
    Arguments
    ---------
    path (str)
        Path to the hdf5 file.
    filters (list[str])
        List of filters to include in the SED.
    phot_path (str)
        Initial path element to reach the filters group.
    conversion (float)
        Multiplicative conversion from catalogue units to output units.
        
    Returns
    -------
    seds (numpy.ndarray)
        2D array where each row corresponds to a single SED.
    """

    # Add the flux in each band to the SED.
    seds = []
    with h5py.File(path, 'r') as f:

        for filter in filters:
            seds.append(f[f'{phot_path}/{filter}'][:] * conversion)

        # Also return the master file indices.
        indices = f['Galaxies/MasterRegionIndex'][:]

    # Get an array of SEDs in the correct format.
    seds = np.array(seds).T

    return seds, indices

def get_progenitors(reg, idx, home_snap, progen_snaps, master_path):
    """
    Extract indices of progenitors and decendents of a galaxy given its
    region and master file index.
    
    Arguments
    ---------
    reg (str)
        Region in which to search.
    idx (int)
        Master file index of target galaxy.
    home_snap (str)
        Snapshot in which target galaxy was identified.
    progen_snaps (List[str])
        Snapshots in which to identify progenitors and decendents.
    master_path (str)
        Path to the FLARES master file.
        
    Returns
    -------
    indices (List[int])
        Master file index of each identified progenitor and decendent
    s_matches (List[int])
        The number of star particles the target and match have in common.
    """

    if isinstance(progen_snaps, str):
        progen_snaps = list(progen_snaps)

    with h5py.File(master_path) as master:
        path_home = f'{reg}/{home_snap}'

        # Get star particle IDs in home snapshot.
        s_length = master[f"{path_home}/Galaxy/S_Length"][:]
        s_start = np.insert(np.cumsum(s_length), 0, 0)
        home_sids = master[f"{path_home}/Particle/S_ID"][s_start[idx]:s_start[idx+1]].astype(np.int64)

        # For each additional snapshot.
        indices = []
        s_matches = []
        for progen_snap in progen_snaps:
            path_progen = f'{reg}/{progen_snap}'

            # Get the corresponding IDs.
            progen_s_length = master[f"{path_progen}/Galaxy/S_Length"][:]
            progen_s_start = np.insert(np.cumsum(progen_s_length), 0, 0)

            # Check which galaxy shares most of these IDs.
            max_matches = 0
            progenitor_index = -1

            for i in range(len(progen_s_length)):

                prog_sids = master[f"{path_progen}/Particle/S_ID"][progen_s_start[i]:progen_s_start[i+1]].astype(np.int64)
                matches = np.isin(prog_sids, home_sids).sum()

                if matches > max_matches:
                    max_matches = matches
                    progenitor_index = i
            
            indices.append(progenitor_index)
            s_matches.append(max_matches)

    return indices, s_matches

def measure_t50(master_path, region, index, snap='008_z007p000'):
    """
    Measure the t50, the time taken to form 50% of the stellar mass.
    
    Arguments
    ---------
    master_path (str)
        Path to the FLARES master file.
    region (str)
        Region of the target galaxy.
    index (int)
        Master file index of target galaxy
    snap (str)
        Snapshot in which target galaxy was identified.

    Returns
    -------
    age_50 (float)
        The age in Myr at which 50% of the stellar mass was formed.
    """

    with h5py.File(master_path, 'r') as master:

        # Stellar length array.
        slength = master[f"{region}/{snap}/Galaxy/S_Length"][:]

        # Get the start indices
        sstart = np.cumsum(slength)
        sstart = np.insert(sstart, 0, 0)

        # Extract the masses and ages.
        p_masses = master[f"{region}/{snap}/Particle/S_Mass"][sstart[index] : sstart[index + 1]] * 1e10
        p_ages = master[f"{region}/{snap}/Particle/S_Age"][sstart[index] : sstart[index + 1]] * 1e3

        # Compute cumulative mass fraction
        sort_indices = np.argsort(p_ages)
        p_ages_sorted = p_ages[sort_indices]
        p_masses_sorted = p_masses[sort_indices]

        cum_mass = np.cumsum(p_masses_sorted)

        # Normalise to total mass.
        cum_mass_frac = cum_mass / cum_mass[-1]

        # Interpolate to find a smoothed t50.
        interp_func = interp1d(cum_mass_frac, p_ages_sorted, kind='linear', bounds_error=False)
        age_50 = interp_func(0.5)

        return age_50