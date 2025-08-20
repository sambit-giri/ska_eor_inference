import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import Planck18 as cos
from astropy import units, constants
# For power spectrum calculations
import tools21cm as t2c

seed = 123456 # Choose your favorite number
np.random.seed(seed)

h = 0.6774

# Folder containing the simulations
# ddir = '/data/cluster/agorce/SKA_chapter_simulations/'
ddir = '../../../data/' # This folder can be created inside the repository folder. It will be ignored during the git commit.
# File with fiducial lightcone
file = ddir+'Lightcone_FID_400_Samples.h5'

# Read h5py file to obtain metadata
with h5py.File(file, 'r') as f:
    frequencies = f['frequencies'][...]  # frequencies along the lightcone
    redshifts = f['redshifts'][...]  # redshifts along the lightcone
    box_length = float(f['box_length'][0])/h  # Mpc
    ngrid = int(f['ngrid'][0])  # number of pixels along the sky patch
    nrand = int(f['nrealisations'][0])  # number of realisations for a given parameter set
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}, mean redshift {np.mean(redshifts):.2f}.')

i = np.random.randint(nrand)


# Read h5py file for slice of lightcone iz-th redshift and i-th realisation
ix = np.random.randint(ngrid)

with h5py.File(file, 'r') as f:
    #full_lc = f['brightness_lightcone'][i, :, :, :]
    lc = f['brightness_lightcone'][i, :, ix, :]  # Reads only (nz, n, n), not entire data

fig, ax = plt.subplots(2, 1, figsize=(12, 12))
im = ax[0].imshow(
    lc,
    extent=(0, box_length, redshifts.min(), redshifts.max()),
    origin='lower', cmap='RdBu_r', aspect='auto'
    )
#ax[0].set_xlabel(r'Resdshift $z$')
#ax[0].set_ylabel(r'$x$ [Mpc]')
fig.colorbar(im, ax=ax[0], label=r'$\delta T_b$ [mK]')
#fig.tight_layout()



nfreq_full = nfreq
red_factor = 4
nfreq = int(nfreq_full/red_factor) # averaging over 8 channels

nu = np.array([np.mean(frequencies[i:i+red_factor]) for i in range(0,nfreq_full, red_factor)])
zz = 1420.406/nu - 1.

lc_ = np.array([np.mean(lc[i:i+red_factor, :], axis=0) for i in range(0,nfreq_full,red_factor)])

im_ = ax[1].imshow(
    lc_,
    extent=(0, box_length, redshifts.min(), redshifts.max()),
    origin='lower', cmap='RdBu_r', aspect='auto'
    )
#ax[0].set_xlabel(r'Resdshift $z$')
#ax[0].set_ylabel(r'$x$ [Mpc]')
fig.colorbar(im_, ax=ax[1], label=r'$\delta T_b$ [mK]')

plt.savefig('maps.png', format='png', dpi=300, bbox_inches='tight')
plt.close()