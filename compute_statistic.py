# Author: Ian Hothi & Adélie Gorce
# Date: Feb 2025

# This script shows how to compute a statistic from the Fisher dataset.
# This example uses the spherical power spectrum.
# Computed statistics are saved in the same h5py file as the input data.

import numpy as np
import h5py
import tqdm
import glob
import os
import gc
# For Power Spectrum Calculations 
import tools21cm as t2c

# Directory where the data is stored
ddir = '/data/cluster/agorce/SKA_chapter_simulations/'

# Global parameters
# Read one h5py file to obtain metadata on simulations
print('Obtaining metadata from file...')
file = ddir+'Lightcone_FID_400_Samples.h5'
with h5py.File(file, 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])  # Mpc
    box_dim = int(f['ngrid'][0])
    n_samp = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')

# statistic params
statname = 'ps'
nbins = 15  # number of k-bins for the spherical ps

# Statistics estimation

# List of simulation files to loop over
files = np.sort(glob.glob(ddir+'Lightcone*h5'))

for fname in files:
    print(f'\nProcessing {os.path.basename(fname)}…')

    # Prepare output container
    PS = np.zeros((n_samp, nbins), dtype=np.float32)
    ks = None

    # Loop over each realisation
    for i in tqdm.tqdm(range(n_samp)):
        with h5py.File(fname, 'r') as f:
            data = f['brightness_lightcone'][i]

        # compute your statistic from the data
        PS[i], ks = t2c.power_spectrum_1d(
            data,
            kbins=nbins,
            box_dims=box_length
        )

    with h5py.File(fname, 'r+') as f:
        f.create_dataset(statname, data=PS, shape=PS.shape)
        f.create_dataset('bins', data=ks, shape=ks.shape)
    print(f'Saved.')

    # Teardown to free memory
    del data, PS
    gc.collect()

print('\nDone.')
