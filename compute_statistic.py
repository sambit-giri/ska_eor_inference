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
# For noise calculations
import tools21cm as t2c
# For stat computation
from scipy.stats import skew, kurtosis

# cosmology
h = 0.6774

# Directory where the data is stored
ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
# ddir = './SKA_chapter_simulations/' # This folder can be created inside the repository folder. It will be ignored during the git commit.

# Overwriting existing statistic
overwrite = True

# Number of CPUs to parallelise over for noise generation
njobs = 4

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

# statistic params
# mean, variance, skewness, kurtosis
statname_mom = 'moments'
nstats = 4
# pixel distribution function
statname_pdf = 'pdf'
dr = 6.
bin_edges = np.arange(-51, 64, step=dr)
bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
nbins = bin_centres.size

# SKA obs parameters
# obs_time = 1000.     # total observation hours
obs_time_array = [100., 1000., 1000.]
layout_array = ['AAstar', 'AAstar', 'AA4']
int_time = 10.       # seconds
total_int_time = 6.  # hours per day
declination = -30.0  # declination of the field in degrees
bmax = 2. * units.km  # km

# Statistics estimation

# List of simulation files to loop over
files = np.sort(glob.glob(ddir+'Lightcone*h5'))

for fname in files:
    print(f'\nProcessing {os.path.basename(fname)} …')

    if overwrite:
        compute = True
    else:
        compute = False
        with h5py.File(fname, 'r+') as f:
            # Remove existing datasets if they exist
            for name in [statname_mom+'_clean', statname_pdf+'_clean']:
                if name not in f:
                    compute = True

    if compute:
        # Prepare output container
        moments_clean = np.zeros((n_samp, nstats), dtype=np.float32)
        moments_noise = np.zeros((n_samp, nstats, len(obs_time_array)), dtype=np.float32)
        moments_obs = np.zeros((n_samp, nstats, len(obs_time_array)), dtype=np.float32)
        stat_clean = np.zeros((n_samp, nbins), dtype=np.float32)
        stat_noise = np.zeros((n_samp, nbins, len(obs_time_array)), dtype=np.float32)
        stat_obs = np.zeros((n_samp, nbins, len(obs_time_array)), dtype=np.float32)

        # Loop over each realisation
        for i in tqdm.tqdm(range(n_samp)):
            # load 21cm brightness lightcone
            with h5py.File(fname, 'r') as f:
                data = f['brightness_lightcone'][i]
            # need to move it to the first axis to match t21c
            data = np.moveaxis(data, 0, 2)
            data = t2c.subtract_mean_signal(data, los_axis=2)
            # compute your statistic from the data
            # clean data
            moments_clean[i, :] = [np.mean(data), np.var(data),
                                   skew(data, axis=None), kurtosis(data, axis=None)]
            stat_clean[i, :], _ = np.histogram(data.flatten(), bins=bin_edges, density=True)
            # if ('FID' in fname):
            for j, (obs_time, layout) in enumerate(zip(obs_time_array, layout_array)):
                print(f'Computing noise for {layout} with obs_time = {obs_time} hours')
                noise_lc = t2c.noise_lightcone(
                    ncells=box_dim,
                    zs=redshifts,
                    obs_time=obs_time,
                    total_int_time=total_int_time,
                    int_time=int_time,
                    declination=declination,
                    subarray_type=layout,
                    boxsize=box_length,
                    verbose=False,
                    save_uvmap=f'{ddir}uvmap_{layout}_{int(obs_time)}hrs.h5',  # save uv coverage to re-use for each realisation
                    n_jobs=njobs,  # Time period of recording the data in seconds.
                    checkpoint=16,  # The code write data after checkpoint number of calculations.
                )  # third axis is line of sight
                # observation = cosmological signal + noise
                dt_obs = t2c.smooth_lightcone(
                    lightcone=noise_lc + data,  # Data cube that is to be smoothed
                    z_array=redshifts,  # Redshifts along the lightcone
                    box_size_mpc=box_length,  # Box size in cMpc
                    max_baseline=bmax,     # Maximum baseline of the telescope
                )[0]
                # noisy data
                moments_obs[i, :, j] = [
                    np.mean(dt_obs), np.var(dt_obs),
                    skew(dt_obs, axis=None), kurtosis(dt_obs, axis=None)
                ]
                stat_obs[i, :, j], _ = np.histogram(dt_obs.flatten(), bins=bin_edges, density=True)
                # noise
                moments_noise[i, :, j] = [
                    np.mean(noise_lc), np.var(noise_lc),
                    skew(noise_lc, axis=None), kurtosis(noise_lc, axis=None)
                ]
                stat_noise[i, :, j], _ = np.histogram(noise_lc.flatten(), bins=bin_edges, density=True)
                del noise_lc, dt_obs

        with h5py.File(fname, 'r+') as f:
            # Remove existing datasets if they exist
            for statname in [statname_mom, statname_pdf]:
                if (statname+'_clean' in f) and overwrite:
                    del f[name+'_clean']
                    for obs_time, layout in zip(obs_time_array, layout_array):
                        del f[f'{statname}_obs_{layout}_{int(obs_time)}hrs']
                        del f[f'{statname}_noise_{layout}_{int(obs_time)}hrs']
            # Save the computed statistics
            f.create_dataset(statname_pdf+'_clean', data=stat_clean, shape=stat_clean.shape)
            f.create_dataset(statname_pdf+'_bins', data=bin_centres, shape=bin_centres.shape)
            f.create_dataset(statname_mom+'_clean', data=moments_clean, shape=moments_clean.shape)
            # if 'FID' in fname:
            for j, (obs_time, layout) in enumerate(zip(obs_time_array, layout_array)):
                f.create_dataset(
                    f'{statname_mom}_noise_{layout}_{int(obs_time)}hrs',
                    data=moments_noise[..., j], shape=moments_noise[..., j].shape)
                f.create_dataset(
                    f'{statname_mom}_obs_{layout}_{int(obs_time)}hrs',
                    data=moments_obs[..., j], shape=moments_obs[..., j].shape)
                f.create_dataset(
                    f'{statname_pdf}_noise_{layout}_{int(obs_time)}hrs',
                    data=stat_noise[..., j], shape=stat_noise[..., j].shape)
                f.create_dataset(
                    f'{statname_pdf}_obs_{layout}_{int(obs_time)}hrs',
                    data=stat_obs[..., j], shape=stat_obs[..., j].shape)
        print('Saved.')

        # Delete to free memory
        del data, stat_clean, stat_noise, stat_obs
        gc.collect()

    else:
        print('Data was already present.')
        print('To redo the calculation, set overwrite to True')

print('\nDone.')
