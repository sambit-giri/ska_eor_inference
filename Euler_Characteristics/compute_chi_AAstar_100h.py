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
import estimator

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
statname = 'chi'
nbins = 21  # number of k-bins for the spherical ps
nubins = np.linspace(-4,4,nbins)

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

    # Construct the output filename (same as before)
    base_fname = os.path.basename(fname)
    name_without_ext = os.path.splitext(base_fname)[0]
    new_base_name = name_without_ext.replace('Lightcone', statname, 1)
    output_base_fname = f"{new_base_name}_{subarray_type}_{int(obs_time)}h.h5"
    output_fname = os.path.join(output_dir, output_base_fname)
    
    # Prepare output containers
    ps_clean = np.zeros((n_samp, nbins), dtype=np.float32)
    ps_noise = np.zeros((n_samp, nbins), dtype=np.float32)
    ps_obs = np.zeros((n_samp, nbins), dtype=np.float32)
        
    start_from = 0
    if not overwrite and os.path.exists(output_fname):
        try:
            with h5py.File(output_fname, 'r') as f_check:
                if 'clean' in f_check and '_last_done' in f_check:
                    last_done_idx = int(f_check['_last_done'][0])
                    start_from = last_done_idx + 1
                    
                    if start_from >= n_samp:
                        print(f"Statistics for {os.path.basename(fname)} already complete. Skipping.")
                        print('To redo the calculation, set overwrite to True.')
                        continue # Skip to the next file
                    
                    print(f"Resuming from sample {start_from}. Loading previous results.")
                    # Load existing data into memory
                    ps_clean[:start_from] = f_check['clean'][:start_from]
                    if 'FID' in fname:
                        ps_noise[:start_from] = f_check['noise'][:start_from]
                        ps_obs[:start_from] = f_check['obs'][:start_from]
        except Exception as e:
            print(f"Could not properly read resume data from {output_fname}: {e}. Starting from scratch.")

    # Loop over each realisation from the determined start_from index
    last_completed_index = -1
    for i in tqdm.tqdm(range(start_from, n_samp), initial=start_from, total=n_samp):
        # load 21cm brightness lightcone
        with h5py.File(fname, 'r') as f:
            data = f['brightness_lightcone'][i]
        # need to move it to the first axis to match t21c
        data = np.moveaxis(data, 0, 2)
        # compute your statistic from the data
        # clean data
        ps_clean[i], ks = estimator.EulerCharacteristicCurve(
            data, nbins=nubins, speed_up='torch', verbose=True
        )
        if ('FID' in fname):
            # generate SKA AA* noise
            noise_lc = t2c.noise_lightcone(
                ncells=box_dim, zs=redshifts, obs_time=obs_time,
                total_int_time=total_int_time, int_time=int_time,
                declination=declination, subarray_type=subarray_type,
                boxsize=box_length, verbose=False, save_uvmap=save_uvmap,
                n_jobs=njobs,
            )
            # observation = cosmological signal + noise
            dt_obs = t2c.smooth_lightcone(
                lightcone=noise_lc + t2c.subtract_mean_signal(data, los_axis=2),
                z_array=redshifts, box_size_mpc=box_length, max_baseline=bmax,
            )[0]
            # noisy data
            ps_obs[i], ks = estimator.EulerCharacteristicCurve(
                dt_obs, nbins=nubins, speed_up='torch', verbose=True
            )
            # noise
            ps_noise[i], ks = estimator.EulerCharacteristicCurve(
                noise_lc, nbins=nubins, speed_up='torch', verbose=True
            )
        
        last_completed_index = i
        with h5py.File(output_fname, 'w') as f_out:
            # Create datasets for metadata
            f_out.create_dataset('frequencies', data=frequencies)
            f_out.create_dataset('redshifts', data=redshifts)
            f_out.create_dataset('box_length', data=np.array([box_length*h]))
            f_out.create_dataset('ngrid', data=np.array([box_dim]))
            f_out.create_dataset('nrealisations', data=np.array([n_samp]))                
            # Save the statistics arrays
            f_out.create_dataset('clean', data=ps_clean)
            f_out.create_dataset('bins', data=ks)
            if 'FID' in fname:
                f_out.create_dataset('noise', data=ps_noise)
                f_out.create_dataset('obs', data=ps_obs)
            f_out.create_dataset('_last_done', data=np.array([last_completed_index]))
        print(f'Saved statistics to {output_fname}. Last completed sample: {last_completed_index}.')
    
    # Clean up memory
    del data, ps_clean, ps_noise, ps_obs
    if 'ks' in locals(): del ks
    gc.collect()

print('\nDone.')