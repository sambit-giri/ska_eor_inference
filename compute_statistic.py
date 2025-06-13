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

# Directory where the data is stored
# ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
ddir = './SKA_chapter_simulations/' # This folder can be created inside the repository folder. It will be ignored during the git commit.

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
    box_length = float(f['box_length'][0])  # Mpc
    ngrid  = int(f['ngrid'][0])  # number of pixels along the sky patch
    n_samp = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')

# The physical length along the line-of-sight (LOS) is different from the field-of-view (FoV).
# Below the list box_length_list should be provided to power spectrum calculator of tools21cm to take this into account.
cdists = cos.comoving_distance(redshifts)
box_length_los = (cdists.max()-cdists.min()).value
box_length_list = [box_length, box_length, box_length_los]

# statistic params
statname = 'PdPS'
nbins = 10  # number of k-bins for the PdPS

def estimate_PdPS(lc, kbins, box_length, box_length_los):
    lc_obs = t2c.subtract_mean_signal(lc, los_axis=2)
    lc_sub = lc_obs[:ngrid//2,:ngrid//2,:]
    dt_mean0 = lc_sub.mean()
    ps0, ks0 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, 
                                     box_dims=[box_length/2,box_length/2,box_length_los])
    lc_sub = lc_obs[:ngrid//2,ngrid//2:,:]
    dt_mean1 = lc_sub.mean()
    ps1, ks1 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, 
                                     box_dims=[box_length/2,box_length/2,box_length_los])
    lc_sub = lc_obs[ngrid//2:,:ngrid//2,:]
    dt_mean2 = lc_sub.mean()
    ps2, ks2 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, 
                                     box_dims=[box_length/2,box_length/2,box_length_los])
    lc_sub = lc_obs[ngrid//2:,ngrid//2:,:]
    dt_mean3 = lc_sub.mean()
    ps3, ks3 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, 
                                     box_dims=[box_length/2,box_length/2,box_length_los])
    iBk = (ps0*dt_mean0+ps1*dt_mean1+ps2*dt_mean2+ps3*dt_mean3)/4
    Pk_mean  = (ps0+ps1+ps2+ps3)/4
    var_mean = (dt_mean0**2+dt_mean1**2+dt_mean2**2+dt_mean3**2)/4
    return iBk/Pk_mean/var_mean, ks0

# SKA obs parameters
obs_time = 1000.     # total observation hours
int_time = 10.       # seconds
total_int_time = 6.  # hours per day
declination = -30.0  # declination of the field in degrees
bmax = 2. * units.km # km

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
            for name in [statname+'_clean', 'bins']:
                if name not in f:
                    compute = True

    if compute:
        # Prepare output container
        pdps_clean = np.zeros((n_samp, nbins), dtype=np.float32)
        pdps_noise = np.zeros((n_samp, nbins), dtype=np.float32)
        pdps_obs   = np.zeros((n_samp, nbins), dtype=np.float32)
    
        # Loop over each realisation
        for i in tqdm.tqdm(range(n_samp)):
            # load 21cm brightness lightcone
            with h5py.File(fname, 'r') as f:
                data = f['brightness_lightcone'][i]
            # need to move it to the first axis to match t21c
            data = np.moveaxis(data, 0, 2)
            # compute your statistic from the data
            # clean data
            pdps_clean[i], ks = estimate_PdPS(
                                data, nbins, 
                                box_length, box_length_los
                            )
            
            if ('FID' in fname):
                # generate SKA AA* noise
                noise_lc = t2c.noise_lightcone(
                    ncells=ngrid,
                    zs=redshifts,
                    obs_time=obs_time,
                    total_int_time=total_int_time,
                    int_time=int_time,
                    declination=declination,
                    subarray_type="AAstar",
                    boxsize=box_length,
                    verbose=False,
                    save_uvmap=ddir+'uvmap_AAstar.h5',  # save uv coverage to re-use for each realisation
                    n_jobs=njobs,  # Time period of recording the data in seconds.
                    checkpoint=16,  # The code write data after checkpoint number of calculations.
                )  # third axis is line of sight
                # observation = cosmological signal + noise
                dt_obs = t2c.smooth_lightcone(
                    lightcone=noise_lc + t2c.subtract_mean_signal(data, los_axis=2),  # Data cube that is to be smoothed
                    z_array=redshifts,  # Redshifts along the lightcone
                    box_size_mpc=box_length,  # Box size in cMpc
                    max_baseline=bmax,     # Maximum baseline of the telescope
                )[0]
                # noisy data
                pdps_obs[i], ks = estimate_PdPS(
                                dt_obs, nbins, 
                                box_length, box_length_los
                            )
                # noise
                pdps_noise[i], ks = estimate_PdPS(
                                noise_lc, nbins, 
                                box_length, box_length_los
                            )
    
        with h5py.File(fname, 'r+') as f:
            # Remove existing datasets if they exist
            for name in [statname+'_clean', statname+'_noise', statname+'_obs', 'bins']:
                if name in f and overwrite:
                    del f[name]
            # Save the computed statistics
            f.create_dataset(statname+'_clean', data=pdps_clean, shape=pdps_clean.shape)
            f.create_dataset('bins', data=ks, shape=ks.shape)
            if 'FID' in fname:
                f.create_dataset(statname+'_noise', data=pdps_noise, shape=pdps_noise.shape)
                f.create_dataset(statname+'_obs', data=pdps_obs, shape=pdps_obs.shape)
        print('Saved.')
        # Teardown to free memory
        del data, pdps_clean, pdps_noise, pdps_obs, ks
        gc.collect()
        
    else:
        print('Data was already present.')
        print('To redo the calculation, set overwrite to True')

print('\nDone.')
