import numpy as np
import h5py
import tqdm
import glob
import os
import gc
from astropy import units
from astropy.cosmology import Planck18 as cosmo
import tools21cm as t2c



def to_cbin(path, map):
    N = np.array(map.shape)
    print(N)
    map = map.reshape((N[0] * N[1] * N[2]), order='c')
    with open(path, 'wb') as f:
        N.astype('int32').tofile(f)
        map.astype('float32').tofile(f)
    f.close()

# cosmology
h = 0.6774

# Directory where the data is stored
ddir = '/scratch/leon/data/ska_ch_data/' 
out_directory = '/scratch/leon/data/ska_ch_data/ska_AA_star_1000/Lightcone_FID_400_Samples'


# Read one h5py file to obtain metadata on simulations
file = ddir+'Lightcone_FID_400_Samples.h5'
with h5py.File(file, 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])/h  # Mpc
    box_dim = int(f['ngrid'][0])
    n_samp = int(f['nrealisations'][0])
    f.close()
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')

# The physical length along the line-of-sight (LOS) is different from the field-of-view (FoV).
# Below the list box_length_list should be provided to power spectrum calculator of tools21cm to take this into account.
cdists = cosmo.comoving_distance(redshifts)
box_length_los = (cdists.max()-cdists.min()).value
box_length_list = [box_length, box_length, box_length_los]

# SKA obs parameters
obs_time = 1000.     # total observation hours
int_time = 10.       # seconds
total_int_time = 6.  # hours per day
declination = -30.0  # declination of the field in degrees
bmax = 2. * units.km # km

njobs = 35  # number of parallel jobs to run

data_fiducial = h5py.File(ddir+"Lightcone_FID_400_Samples.h5")
# n_samp = 1
for i in tqdm.tqdm(range(n_samp)):
    data = data_fiducial['brightness_lightcone'][10]  
    data = np.moveaxis(data, 0, 2)
    noise_lc = t2c.noise_lightcone(
                    ncells=box_dim,
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
                )  # third axis is line of sight
    # with_out_smooth = noise_lc + t2c.subtract_mean_signal(data, los_axis=2)  # Data cube that is to be smoothed
    dt_obs = t2c.smooth_lightcone(
                    lightcone=noise_lc + t2c.subtract_mean_signal(data, los_axis=2),  # Data cube that is to be smoothed
                    z_array=redshifts,  # Redshifts along the lightcone
                    box_size_mpc=box_length,  # Box size in cMpc
                    max_baseline=bmax,     # Maximum baseline of the telescope
                )[0]
    to_cbin('{}/realization_{}.cbin'.format(out_directory, int(i)), dt_obs)
