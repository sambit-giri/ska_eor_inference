# Author: Ian Hothi, Adélie Gorce, & Sambit Giri
# Date: Feb 2025
# Author: Edited by Abinash Kumar Shaw (to include MAPS computation)
# Date: July 2025

# This script shows how to compute a statistic from the Fisher dataset,
# first on clean data, then on noisy data for the AA* configuration of SKA.
# This example uses the spherical power spectrum.
# Computed statistics are saved in the same h5py file as the input data.

import numpy as np
import h5py
import tqdm
import glob
import os, sys
import gc
from astropy import units
from astropy.cosmology import Planck18 as cos
from scipy.fft import rfft2, irfft2, fftfreq, rfftfreq
# For Power Spectrum and noise calculations
import tools21cm as t2c
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Function to compute the MAPS statistics given a data volume
def comp_maps(inarr, bin_array, dth=0.1, lbin=10, nfreq=100, area=10, nthreads=1):
    vis_data = rfft2(inarr*(dth**2.), axes=(0, 1), workers=8) # dth^2 is multiplied for the DFT

    bin_maps = np.zeros((lbin, nfreq, nfreq), dtype=np.float64, order='C') # to store the maps data

    for i in range(lbin):
        for j in range(nfreq):
            for k in range(nfreq):
                data1 = vis_data[:, :, j]
                data2 = vis_data[:, :, k]
                data1 = data1[bin_array==i+1]
                data2 = np.conj(data2[bin_array==i+1]) # complexx conjugate is done here
                bin_maps[i, j, k] = np.real(np.mean(data1 * data2)) / area # taking only the real part

    return bin_maps
##########################################################

# cosmology
h = 0.6774

# Directory where the data is stored
# ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
ddir = '../../data/' # This folder can be created inside the repository folder. It will be ignored during the git commit.

# Overwriting existing statistic
overwrite = True #False

# Number of CPUs to parallelise over for noise generation
njobs = 16 #4

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
r_z = cdists.value
box_length_los = (cdists.max()-cdists.min()).value
box_length_list = [box_length, box_length, box_length_los]

# statistic params
statname = 'mapsAA4'
nbins = 10  # number of k-bins for the spherical ps

################ MAPS grid parameters ####################
Nz = len(redshifts)
ang_boxsize = float(box_length / r_z[Nz//2]) # angular size of the box face at central redshift in rad
dL = box_length/ box_dim # size of each voxel in Mpc
dtheta = float(dL/ r_z[Nz//2])
Dl = 2.* np.pi/ang_boxsize # grid size in (l,m) space
lmax = Dl * 0.5 * box_dim # max of (l,m) range withing Nyquist range
Omega = ang_boxsize**2. # solid angle in rad^2

lx = 2.* np.pi * fftfreq(box_dim, d=ang_boxsize/box_dim)
ly = 2.* np.pi* rfftfreq(box_dim, d=ang_boxsize/box_dim)
#print(lx)
#print(ly)
lxgrid, lygrid = np.meshgrid(ly, lx)
lgrid = np.sqrt(lxgrid**2. + lygrid**2.)
#print(lgrid)
bins = np.logspace(np.log10(Dl), np.log10(lmax), num=nbins+1, endpoint=True)
bindex = np.digitize(lgrid, bins, right=False)
Nell = np.array([np.sum(bindex==i) for i in range(1, nbins+1)]) # stores the number of ell grids in a bin
ell = np.array([np.mean(lgrid[bindex==i]) for i in range(1, nbins+1)]) # avg ell grids within a bin
##############################################################

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
        maps_clean = np.zeros((n_samp, nbins, nfreq, nfreq), dtype=np.float32, order='C')
        maps_noise = np.zeros((n_samp, nbins, nfreq, nfreq), dtype=np.float32, order='C')
        maps_obs = np.zeros((n_samp, nbins, nfreq, nfreq), dtype=np.float32, order='C')
    
        # Loop over each realisation
        for i in tqdm.tqdm(range(n_samp)):
            # load 21cm brightness lightcone
            with h5py.File(fname, 'r') as f:
                data = f['brightness_lightcone'][i]
            # need to move it to the first axis to match t21c
            data = np.moveaxis(data, 0, 2)

            # compute your statistic from the data
            # clean data

            data = t2c.subtract_mean_signal(data, los_axis=2) # subtracting the mean of the data

            maps_clean[i, :, :, :] = comp_maps(data, bindex, dth=dtheta, lbin=nbins, nfreq=nfreq, area=Omega, nthreads=njobs) * (ell[:, None, None] * (ell[:, None, None] +1.)/2./np.pi)
            
            """
            plt.imshow(maps_res[1, :, :] , origin='lower', interpolation=None)
            plt.colorbar()
            plt.savefig('maps_2.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            sys.exit(0)


            ps_clean[i], ks = t2c.power_spectrum_1d(
                data,
                kbins=nbins,
                box_dims=box_length_list
            )
            """
            if ('FID' in fname):
                # generate SKA AA* noise
                noise_lc = t2c.noise_lightcone(
                    ncells=box_dim,
                    zs=redshifts,
                    obs_time=obs_time,
                    total_int_time=total_int_time,
                    int_time=int_time,
                    declination=declination,
                    subarray_type="AA4",
                    boxsize=box_length,
                    verbose=False,
                    save_uvmap=ddir+'uvmap_AA4.h5',  # save uv coverage to re-use for each realisation
                    n_jobs=njobs,  # Time period of recording the data in seconds.
                    checkpoint=16,  # The code write data after checkpoint number of calculations.
                )  # third axis is line of sight
                # observation = cosmological signal + noise
                dt_obs = noise_lc + data # I am not smooting it (Abinash)
                """
                t2c.smooth_lightcone(
                    lightcone=noise_lc + data, #t2c.subtract_mean_signal(data, los_axis=2),  # Data cube that is to be smoothed
                    z_array=redshifts,  # Redshifts along the lightcone
                    box_size_mpc=box_length,  # Box size in cMpc
                    max_baseline=bmax,     # Maximum baseline of the telescope
                )[0]
                
                # noisy data
                
                ps_obs[i], ks = t2c.power_spectrum_1d(
                    dt_obs,
                    kbins=nbins,
                    box_dims=box_length_list
                )
                # noise
                ps_noise[i], ks = t2c.power_spectrum_1d(
                    noise_lc,
                    kbins=nbins,
                    box_dims=box_length_list
                )
                """
                maps_obs[i, :, :, :] = comp_maps(dt_obs, bindex, dth=dtheta, lbin=nbins, nfreq=nfreq, area=Omega, nthreads=njobs) * (ell[:, None, None] * (ell[:, None, None] +1.)/2./np.pi)
                maps_noise[i, :, :, :] = comp_maps(noise_lc, bindex, dth=dtheta, lbin=nbins, nfreq=nfreq, area=Omega, nthreads=njobs) * (ell[:, None, None] * (ell[:, None, None] +1.)/2./np.pi)

        with h5py.File(fname, 'r+') as f:
            # Remove existing datasets if they exist
            for name in [statname+'_clean', statname+'_noise', statname+'_obs', 'bins']:
                if name in f and overwrite:
                    del f[name]
            # Save the computed statistics
            f.create_dataset(statname+'_clean', data=maps_clean, shape=maps_clean.shape)
            f.create_dataset('bins', data=ell, shape=ell.shape)
            if 'FID' in fname:
                f.create_dataset(statname+'_noise', data=maps_noise, shape=maps_noise.shape)
                f.create_dataset(statname+'_obs', data=maps_obs, shape=maps_obs.shape)
        print('Saved.')
    
        # Teardown to free memory
        del data, maps_clean, maps_noise, maps_obs
        gc.collect()

    else:
        print('Data was already present.')
        print('To redo the calculation, set overwrite to True')

print('\nDone.')

