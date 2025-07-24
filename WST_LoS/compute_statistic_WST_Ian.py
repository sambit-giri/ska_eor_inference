"""
Compute Wavelet Scattering Transform (WST) statistic for different SKA layouts and observing times.
Processes AA* and AA4 configurations, at 100h and 1000h for AA*.
Reads input HDF5 files and writes only the L1/L2 decompositions into output files.
"""
import numpy as np
import h5py
import tqdm
import glob
import os
import gc
import shutil
from astropy import units
from astropy.cosmology import Planck18 as cos
import pywt

# Compatibility shim
try:
    import importlib.resources as _res
    from importlib_resources import files as _b_files
    if not hasattr(_res, 'files'):
        _res.files = _b_files
except ImportError:
    pass
import tools21cm as t2c

# Paths
data_dir = '/loreli/ihothi/SKA_Chapter/Simulations'
out_dir  = '/travail/ihothi/SKA_chapter_simulations'
os.makedirs(out_dir, exist_ok=True)

# Window functions
wf_file = os.path.join('/travail/ihothi/21cmFast_Save/', 'PS_2D_Window_Functions_z8_z9.npz')
with np.load(wf_file, allow_pickle=True) as wf:
    Binned_ft2d_WF = wf['arr_0']

# Parameters
bins_2d = 10
# now only first and second layer: nbins + nbins*(nbins-1)/2
total_scales = int(bins_2d + (bins_2d*(bins_2d-1)//2))
l1l2_scales = [2, 4]
# Observation configs: (layout, hours, suffix)
configs = [
    ("AAstar", 1000., "aastar1000"),
    ("AAstar", 100.,  "aastar100"),
    ("AA4",    1000., "aa4")
]
# Shared SKA noise settings
int_time       = 10.    # seconds
total_int_time = 6.     # hours/day
declination    = -30.0  # degrees
bmax           = 2. * units.km  # baseline only used for smoothing
njobs          = 1      # parallel jobs

# WST per-slice (omit zeroth moment)
def New_WST_Load(dat, nbins, Binned_ft2d_WF):
    from numpy import fft as f
    small2large = Binned_ft2d_WF[::-1]
    # first layer
    area = dat.size
    ft2d = f.fftn(dat) / area
    ft2d = f.fftshift(ft2d)
    S1 = np.zeros(nbins)
    for i in range(nbins):
        conv1 = f.ifftn(f.ifftshift(ft2d * small2large[i]))
        S1[i] = np.sum(np.abs(conv1))
    # second layer normalised
    n2 = nbins*(nbins-1)//2
    S2_norm = np.zeros(n2)
    idx = 0
    for i in range(nbins):
        conv1_abs = np.abs(f.ifftn(f.ifftshift(ft2d * small2large[i])))
        ft_c1 = f.fftshift(f.fftn(conv1_abs) / area)
        for j in range(i+1, nbins):
            conv2 = f.ifftn(f.ifftshift(ft_c1 * small2large[j]))
            S2_norm[idx] = np.sum(np.abs(conv2)) / S1[i]
            idx += 1
    # combine first & second layers
    combined = np.zeros(nbins + n2)
    combined[0:nbins] = S1
    combined[nbins:] = S2_norm
    return combined

# LoS L1/L2
def LoS_Decomp_L1L2(Data, coeffs):
    out = np.zeros((Data.shape[0], coeffs, 2*len(l1l2_scales)+1))
    for j in range(Data.shape[0]):
        for k in range(coeffs):
            evo = Data[j, :, k]
            cwtmatr, _ = pywt.cwt(evo, l1l2_scales, 'morl')
            for l, row in enumerate(cwtmatr):
                out[j, k, 2*l]   = np.linalg.norm(row)
                out[j, k, 2*l+1] = np.sum(np.abs(row))
            out[j, k, -1] = np.mean(evo)
    return out

# Loop input files
files = sorted(glob.glob(os.path.join(data_dir, 'Lightcone*.h5')))
for src in files:
    base = os.path.basename(src)
    dst = os.path.join(out_dir, base)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    print(f"Processing {base} …")
    # read metadata
    with h5py.File(dst, 'r+') as f:
        freqs     = f['frequencies'][...]
        redshifts = f['redshifts'][...]
        box_len   = float(f['box_length'][0]) / cos.h
        box_dim   = int(f['ngrid'][0])
        nreal     = int(f['nrealisations'][0])
    nfreq = freqs.size

    # containers for FID + each config
    wsts = {'fid': np.zeros((nreal, nfreq, total_scales), dtype=np.float32)}
    for _, _, suf in configs:
        wsts[suf] = np.zeros((nreal, nfreq, total_scales), dtype=np.float32)

    # process lightcone realisations
    for i in tqdm.tqdm(range(nreal), desc='Realizations'):
        np.random.seed(i)
        with h5py.File(src, 'r') as f:
            lc = f['brightness_lightcone'][i]
        data = np.moveaxis(lc, 0, 2)

        # compute clean (FID) WST
        for k in range(nfreq):
            wsts['fid'][i, k] = New_WST_Load(data[:, :, k], bins_2d, Binned_ft2d_WF)

        # compute observed WST for each config
        for layout, hours, suf in configs:
            noise_lc = t2c.noise_lightcone(
                ncells=box_dim, zs=redshifts,
                obs_time=hours, total_int_time=total_int_time,
                int_time=int_time, declination=declination,
                subarray_type=layout, boxsize=box_len,
                verbose=False, save_uvmap=os.path.join(out_dir, f'uvmap_{layout}.h5'),
                n_jobs=njobs
            )
            dt_obs = t2c.smooth_lightcone(
                lightcone=noise_lc + t2c.subtract_mean_signal(data, los_axis=2),
                z_array=redshifts, box_size_mpc=box_len,
                max_baseline=bmax
            )[0]
            for k in range(nfreq):
                wsts[suf][i, k] = New_WST_Load(dt_obs[:, :, k], bins_2d, Binned_ft2d_WF)

    # apply LoS L1/L2
    wst_l1l2 = {suf: LoS_Decomp_L1L2(w, total_scales)
                for suf, w in wsts.items()}

    # Save only the L1/L2 results for all cases
    with h5py.File(dst, 'r+') as f:
        for suf, arr in wst_l1l2.items():
            name = f'wst_{suf}_l1l2'
            if name in f:
                del f[name]
            f.create_dataset(name, data=arr)
    gc.collect()
