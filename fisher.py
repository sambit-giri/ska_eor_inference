import numpy as np
import h5py
import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import SymLogNorm   
import corner
import os, sys
import glob

# Folder where the data is stored
# ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
ddir = '../../../data/' # This folder can be created inside the repository folder. It will be ignored during the git commit.
# File with fiducial lightcone
file = ddir+'Lightcone_FID_400_Samples.h5'

statname = 'smt_redmapsAA4'

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


nfreq_full = nfreq
red_factor = 8
nfreq = int(nfreq_full/red_factor)

nu = np.array([np.mean(frequencies[i:i+red_factor]) for i in range(0,nfreq_full, red_factor)])
zz = 1420.406/nu - 1.

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


"""
clean = MAPS_data['FID']['maps_clean'][:]
obs = MAPS_data['FID']['maps_obs'][:]
noise = MAPS_data['FID']['maps_noise'][:]

clean_diag = np.diagonal(clean, axis1=2, axis2=3)
obs_diag = np.diagonal(obs, axis1=2, axis2=3)
noise_diag = np.diagonal(noise, axis1=2, axis2=3)

mean_clean = np.mean(clean_diag, axis=0)
mean_obs = np.mean(obs_diag, axis=0)
mean_noise = np.mean(noise_diag, axis=0)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharex=True, sharey=False, gridspec_kw=dict(hspace=0.2, wspace=0.2)) 

binnum = 9

for i in range(0,n_samp,40):
    ax[0].semilogy(clean_diag[i, binnum, :], lw=1, alpha=0.4)
    ax[1].semilogy(obs_diag[i, binnum, :], lw=1, alpha=0.4)
    ax[2].semilogy(noise_diag[i, binnum, :], lw=1, alpha=0.4)
    #break

ax[0].plot(mean_clean[binnum,:], lw=2, ls='--', c='k')
ax[1].plot(mean_obs[binnum,:], lw=2, ls='--', c='k')
ax[2].plot(mean_noise[binnum,:], lw=2, ls='--', c='k')

ax[0].set_title(r'Clean')
ax[1].set_title(r'Obs')
ax[2].set_title(r'Noise')

plt.savefig(f'diag_ell{binnum}.png', format='png', dpi=300, bbox_inches='tight')
plt.close()

sys.exit(0)
"""

# Now computing the covariance of the MAPS using 400 realizations of the signal

fid_data =  MAPS_data['FID']['maps_obs'][:]
ell = MAPS_data['FID']['ell'][:]


ell_lim = 8

fid_data = fid_data[:, :ell_lim, :, :]
ell = ell[:ell_lim]
MAPS_derivs = MAPS_derivs[:, :, :ell_lim, : , :]

Nell = len(ell)

stdx = np.std(fid_data, axis=0)

fid_data = fid_data/stdx[None,:,:,:]
MAPS_derivs = MAPS_derivs/stdx[None, None,:,:,:]

F = np.zeros((nparams, nparams), dtype=np.float64, order='C')

condition = np.zeros((nfreq, nfreq), dtype=np.float64, order='C')

for i in range(nfreq):
    for j in range(nfreq):
        derv_ = np.mean(MAPS_derivs[:,:,:,i,j], axis=1)
        data_ = fid_data[:,:,i,j]
        errcov = np.cov(data_, rowvar=False)
        condition[i,j] = np.log10(np.linalg.cond(errcov))
        inv_cov = np.linalg.inv(errcov)
        x = np.matmul(derv_, inv_cov)
        #print(x.shape)
        F += np.matmul(x, derv_.T)

im = plt.imshow(condition, origin='lower', cmap='coolwarm')
plt.colorbar(im)
plt.xlabel(r'$\nu_1$')
plt.ylabel(r'$\nu_2$')
plt.title(r'Condition number')
plt.gcf().set_size_inches(7,6)
plt.savefig('condition_AA4_1000.png', format='png', dpi=300, bbox_inches='tight')
plt.clf()
plt.close()

param_cov = np.linalg.inv(F)
print(param_cov)

np.savetxt('param_cov_1000_AA4.txt', param_cov, fmt='%.6e', delimiter='\t', newline='\n')

# Corner plot
"""
fisher_data = np.random.multivariate_normal(fid, param_cov, size=100000)
fig = corner.corner(
    fisher_data,
    labels=Fisher_Param,
    plot_datapoints=False,  
    levels=(0.68,0.95),
    truths=fid)
fig.savefig('corner_diag_1000_AA4.png', format='png', dpi=300, bbox_inches='tight')
plt.clf()
plt.close()

sys.exit(0)
"""
############### Checking the convergence of the Fisher analysis ################
# Number of realisations to look at when checking convergence
samples = np.arange(20, n_samp+5, 5)

# Initialising Fishers 
MAPS_Fisher = np.zeros((samples.size, nparams, nparams))
MAPS_Fisher_Inv = np.zeros((samples.size, nparams, nparams))

# Calculating the Fisher matrix for a given sample of derivatives 
for k, sample_size in enumerate(samples):
    for i in range(nfreq):
        for j in range(nfreq):
            derv_ = np.mean(MAPS_derivs[:,:sample_size,:,i,j], axis=1)
            data_ = fid_data[:sample_size,:,i,j]
            errcov = np.cov(data_, rowvar=False)
            inv_cov = np.linalg.inv(errcov)
            x = np.matmul(derv_, inv_cov)
            #print(x.shape)
            MAPS_Fisher[k] += np.matmul(x, derv_.T)
    MAPS_Fisher_Inv[k] = np.linalg.inv(MAPS_Fisher[k])

plt.figure(figsize=(12, 4))
for i, param in enumerate(params):
    plt.plot(samples, [np.diagonal(MAPS_Fisher_Inv[k])[i]-np.diagonal(MAPS_Fisher_Inv[-1])[i] for k in range(samples.size)], marker='.', label=param)
plt.legend()
plt.ylabel(r'$\sigma^2_{ii}$')
plt.xlabel('Number of samples')
plt.title('Convergence of the Parameters Variance')
plt.savefig('convg_1000_AA4.png', format='png', dpi=300, bbox_inches='tight')