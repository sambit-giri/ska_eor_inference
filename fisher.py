# Author: Abinash Kumar Shaw
# Date: July 2025

# This script shows how to compute Fisher information matrix from computed MAPS statistics.
# Change the configuration and the statsname to accommodate for AAstar (100 and 1000 hrs)
# and AA4 (1000 hrs) error covariances.


import numpy as np
import h5py
import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import corner
import os, sys
import glob

# Folder where the data is stored
# ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
ddir = '../../data/' # This folder can be created inside the repository folder. It will be ignored during the git commit.
# File with fiducial lightcone
file = ddir+'Lightcone_FID_400_Samples.h5'

statname = 'mapsAA4' # change the statsname here

# cosmology
h = 0.6774

# Astro params
params = ['ION_Tvir_MIN','R_BUBBLE_MAX','HII_EFF_FACTOR']
Fisher_Param = ['$T_{Vir}$','$R_{Max}$','$\zeta$']
nparams = len(params)
# parameter values for fiducial simulation
fid = [pow(10,4.7),15,30]
# parameter range for derivatives
delta_params = [pow(10,4.740362689494244)-pow(10,4.653212513775344), 10, 10]

# Power spectrum params
nbins = 10

# simulation files
files = [ddir+'Lightcone_FID_400_Samples.h5']
[files.extend(
    [f'{ddir}Lightcone_{p}_400_Samples_Plus.h5', f'{ddir}Lightcone_{p}_400_Samples_Minus.h5']) for p in params]
#print(files)

# Read h5py file for metadata
with h5py.File(files[0], 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])/h  # Mpc
    box_dim = int(f['ngrid'][0])
    n_samp = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')



# container dict: keys will be e.g. 'PS_param_1_plus', 'PS_fid', etc.
MAPS_data = {}
keys = []
count = 0
for fpath in tqdm.tqdm(files):

    key = os.path.basename(fpath)[:-3].replace('Lightcone_', '').replace('_400_Samples', '')
    keys.append(key)
    try:
        with h5py.File(fpath, 'r') as f:
            ell = f['bins'][...]
            #mask = ks < k_nyquist
            MAPS_data[key] = {
                'maps_clean':  f[statname+'_clean'][:],  # ps of clean cosmological signal
                'ell':  f['bins'][:],
            }
            if 'FID' in key:
                MAPS_data[key].update({
                    'maps_noise':  f[statname+'_noise'][:],  # ps of noise
                    'maps_obs':  f[statname+'_obs'][:],  # ps of smoothed (noise + cosmological signal)
                })
    except KeyError:
        MAPS_data[key] = {}
        MAPS_data[key]['maps_clean'] = np.zeros((n_samp, nbins, nfreq, nfreq))
        MAPS_data[key]['maps_noise'] = np.zeros((n_samp, nbins, nfreq, nfreq))
        MAPS_data[key]['maps_obs'] = np.zeros((n_samp, nbins, nfreq, nfreq))
        MAPS_data[key]['ell'] = np.zeros(nbins)
        count += 1
#print(keys)
# now you can refer to, e.g. PS_data['PS_R_BUBBLE_MAX_plus']['PS'] 
# and PS_data['PS_R_BUBBLE_MAX_plus']['ks'] etc.
print(f'Found {len(MAPS_data)} {statname} datasets, {count} missing.')

# Example: check shapes
for name, dat in MAPS_data.items():
    print(f"{name:25s} → MAPS {dat['maps_clean'].shape},  ell {dat['ell'].shape}")



# Compute the finite‐difference derivatives for each k‐bin & sample
# from the statistics derived from the clean signal
dMAPS = {}
MAPS_derivs = np.zeros((len(params), n_samp, nbins, nfreq, nfreq))
for ip, param in enumerate(params):
    dMAPS[param] = (MAPS_data[f'{param}_Plus']['maps_clean'] - MAPS_data[f'{param}_Minus']['maps_clean']) / delta_params[ip]
    MAPS_derivs[ip] = (MAPS_data[f'{param}_Plus']['maps_clean'] - MAPS_data[f'{param}_Minus']['maps_clean']) / delta_params[ip]
#ks = MAPS_data[f'{param}_Plus']['ks']

print(dMAPS['R_BUBBLE_MAX'].shape)
print(MAPS_derivs.shape)

# Computing the mean derivatives
avg_derv = np.mean(MAPS_derivs, axis=1) # computing the average of the derivatives over 400 realizations
derv_avg = np.reshape(avg_derv, shape=(3, 10*128*128), order='C')

# Now computing the covariance of the MAPS using 400 realizations of the signal

data =  MAPS_data['FID']['maps_obs'][:]
estim_data = np.reshape(data, shape=(400, 10*128*128), order='C') # reshaping
sig = np.std(estim_data, axis=0) # computing the error variance of the data

x = derv_avg/ sig[None,:]
print(x.shape)

# computing Fisher Matrix assuming uncorrelated modes and frequency (Gaussian approximation)
F = np.matmul(x, x.T)
print(F.shape)

param_cov = np.linalg.inv(F) # parameter error covariance
print(param_cov)

np.savetxt('paramcov_1000_AA4.txt', param_cov, fmt='%e', delimiter='\t', newline='\n') # change the filename here

# Corner plot
fisher_data = np.random.multivariate_normal(fid, param_cov, size=100000)
fig = corner.corner(
    fisher_data,
    labels=Fisher_Param,
    plot_datapoints=False,  
    levels=(0.68,0.95),
    truths=fid)
fig.savefig('corner_1000_AA4.png', format='png', dpi=300, bbox_inches='tight') # change the filename here
plt.clf()
plt.close()

############### Checking the convergence of the Fisher analysis ################
# Number of realisations to look at when checking convergence
samples = np.arange(5, n_samp+5, 5)

flat_derivs = np.reshape(MAPS_derivs, shape=(3, 400, 10*128*128), order='C')

# Initialising Fishers 
MAPS_Fisher = np.zeros((samples.size, nparams, nparams))
MAPS_Fisher_Inv = np.zeros((samples.size, nparams, nparams))

# Calculating the Fisher matrix for a given sample of derivatives 
for k, sample_size in enumerate(samples):
    deriv_sample = np.mean(flat_derivs[:, :sample_size, :], axis=1)
    cov_sample = np.std(estim_data[:sample_size, :], axis=0)
    x = deriv_sample/ cov_sample[None, :]
    MAPS_Fisher[k] = np.matmul(x, x.T)
    MAPS_Fisher_Inv[k] = np.linalg.inv(MAPS_Fisher[k])

plt.figure(figsize=(12, 4))
for i, param in enumerate(params):
    plt.plot(samples, [np.diagonal(MAPS_Fisher_Inv[k])[i]-np.diagonal(MAPS_Fisher_Inv[-1])[i] for k in range(samples.size)], marker='.', label=param)
plt.legend()
plt.ylabel(r'$\sigma^2_{ii}$')
plt.xlabel('Number of samples')
plt.title('Convergence of the Parameters Variance')
plt.savefig('convg_1000_AA4.png', format='png', dpi=300, bbox_inches='tight') # change filename here