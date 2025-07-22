# Author: Ian Hothi, Adélie Gorce, & Sambit Giri
# Date: Feb 2025

# This script shows how to compute a statistic from the Fisher dataset,
# first on clean data, then on noisy data for the AA* configuration of SKA.
# This example uses the spherical power spectrum.
# Computed statistics are saved in the same h5py file as the input data.

import numpy as np
import h5py
import tqdm
import glob
import os
import gc
from astropy import units
from astropy.cosmology import Planck18 as cos
# For Power Spectrum and noise calculations
import tools21cm as t2c

# cosmology
h = 0.6774

# Directory where the data is stored
# ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
ddir = '../SKA_chapter_simulations/' # This folder can be created inside the repository folder. It will be ignored during the git commit.
output_dir = './SKA_chapter_statistics/'

# Overwriting existing statistic
overwrite = False #True

# Number of CPUs to parallelise over for noise generation
njobs = 1 #4

# Global parameters
# Read one h5py file to obtain metadata on simulations
print('Obtaining metadata from file...')
file = ddir+'Lightcone_FID_400_Samples.h5'
with h5py.File(file, 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])/h  # Mpc
    box_dim = int(f['ngrid'][0])
    n_samp = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')

# The physical length along the line-of-sight (LOS) is different from the field-of-view (FoV).
# Below the list box_length_list should be provided to power spectrum calculator of tools21cm to take this into account.
cdists = cos.comoving_distance(redshifts)
box_length_los = (cdists.max()-cdists.min()).value
box_length_list = [box_length, box_length, box_length_los]

# Statistic parameters
statname = 'ps' # Name of the statistic (e.g., 'ps' for power spectrum)
nbins = 15  # Number of k-bins for the spherical power spectrum

# SKA obs parameters
obs_time = 100.      # total observation hours
int_time = 10.       # seconds
total_int_time = 6.  # hours per day
declination = -30.0  # declination of the field in degrees
bmax = 2. * units.km # km
subarray_type = "AAstar" # Type of subarray for noise generation (e.g., "AAstar", "AA4")
save_uvmap = ddir+'uvmap_AAstar.h5' # save uv coverage to re-use for each realisation

# Statistics estimation

# List of simulation files to loop over
files = np.sort(glob.glob(ddir+'Lightcone*h5'))

for fname in files:
    print(f'\nProcessing {os.path.basename(fname)} …')

    # Construct the output filename for the current input file
    base_fname = os.path.basename(fname)
    # Get filename without extension
    name_without_ext = os.path.splitext(base_fname)[0]
    # Replace 'Lightcone' with the chosen statistic name (e.g., 'ps')
    new_base_name = name_without_ext.replace('Lightcone', statname, 1) # Replace only the first occurrence
    output_base_fname = f"{new_base_name}_{subarray_type}_{int(obs_time)}h.h5" 
    output_fname = os.path.join(output_dir, output_base_fname)

    # Determine if computation is needed
    compute = False
    if overwrite or not os.path.exists(output_fname):
        compute = True # Compute if overwrite is True or output file does not exist
    else:
        # If output file exists and not overwriting, check if essential datasets are present
        try:
            with h5py.File(output_fname, 'r') as f_out_check:
                # Check for 'clean' and 'bins' datasets
                if 'clean' not in f_out_check or 'bins' not in f_out_check:
                    compute = True # Recompute if clean PS or k-bins are missing
                # Check for 'noise' and 'obs' datasets only for 'FID' files
                if 'FID' in fname and ('noise' not in f_out_check or 'obs' not in f_out_check):
                    compute = True # Recompute if FID data and noise/obs PS are missing
        except Exception as e:
            print(f"Error checking existing output file {output_fname}: {e}. Recomputing.")
            compute = True # Recompute if there's an error reading the existing file

    if compute:
        # Prepare output container
        ps_clean = np.zeros((n_samp, nbins), dtype=np.float32)
        ps_noise = np.zeros((n_samp, nbins), dtype=np.float32)
        ps_obs = np.zeros((n_samp, nbins), dtype=np.float32)
    
        # Loop over each realisation
        for i in tqdm.tqdm(range(n_samp)):
            # load 21cm brightness lightcone
            with h5py.File(fname, 'r') as f:
                data = f['brightness_lightcone'][i]
            # need to move it to the first axis to match t21c
            data = np.moveaxis(data, 0, 2)
            # compute your statistic from the data
            # clean data
            ps_clean[i], ks = t2c.power_spectrum_1d(
                data,
                kbins=nbins,
                box_dims=box_length_list
            )
            if ('FID' in fname):
                # generate SKA AA* noise
                noise_lc = t2c.noise_lightcone(
                    ncells=box_dim,
                    zs=redshifts,
                    obs_time=obs_time,
                    total_int_time=total_int_time,
                    int_time=int_time,
                    declination=declination,
                    subarray_type=subarray_type,
                    boxsize=box_length,
                    verbose=False,
                    save_uvmap=save_uvmap,  
                    n_jobs=njobs,  # Time period of recording the data in seconds.
                )  # third axis is line of sight
                # observation = cosmological signal + noise
                dt_obs = t2c.smooth_lightcone(
                    lightcone=noise_lc + t2c.subtract_mean_signal(data, los_axis=2),  # Data cube that is to be smoothed
                    z_array=redshifts,  # Redshifts along the lightcone
                    box_size_mpc=box_length,  # Box size in cMpc
                    max_baseline=bmax,     # Maximum baseline of the telescope
                )[0]
                # noisy data
                ps_obs[i], ks = t2c.power_spectrum_1d( # _ indicates k-bins are discarded as ks is already set
                    dt_obs,
                    kbins=nbins,
                    box_dims=box_length_list
                )
                # noise
                ps_noise[i], ks = t2c.power_spectrum_1d( # _ indicates k-bins are discarded
                    noise_lc,
                    kbins=nbins,
                    box_dims=box_length_list
                )

        # Save the computed statistics to the new output HDF5 file
        with h5py.File(output_fname, 'w') as f_out:
            # Create datasets for metadata
            f_out.create_dataset('frequencies', data=frequencies)
            f_out.create_dataset('redshifts', data=redshifts)
            f_out.create_dataset('box_length', data=np.array([box_length*h])) # Store original Mpc/h value
            f_out.create_dataset('ngrid', data=np.array([box_dim]))
            f_out.create_dataset('nrealisations', data=np.array([n_samp]))
            # Renamed dataset attributes to 'clean', 'noise', 'obs'
            f_out.create_dataset('clean', data=ps_clean, shape=ps_clean.shape)
            f_out.create_dataset('bins', data=ks, shape=ks.shape) # Save k-bins
            if 'FID' in fname: # Only save noise and observed PS for FID files
                f_out.create_dataset('noise', data=ps_noise, shape=ps_noise.shape)
                f_out.create_dataset('obs', data=ps_obs, shape=ps_obs.shape)
        print(f'Saved statistics to {output_fname}.')

        # Clean up memory after processing each file
        del data, ps_clean, ps_noise, ps_obs, ks
        gc.collect()

    else:
        # Message when computation is skipped
        print(f'Statistics for {os.path.basename(fname)} were already present in {output_fname}.')
        print('To redo the calculation, set overwrite to True.')

print('\nDone.')
