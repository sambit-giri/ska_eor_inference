import numpy as np
from astropy import units, constants
# For power spectrum calculations
import tools21cm as t2c
import h5py

def tau_to_kpara(z, cosmo):
    dRpara_df = (1 + z)**2.0 / cosmo.H(z).value * (constants.c.value / 1e3) / 1420405751.0
    return 2 * np.pi / dRpara_df

def bl_to_kperp(z, cosmo):
    # Parsons 2012, Pober 2014, Kohn 2018
    f0 = 1420405751.0 / (z + 1)
    return 2 * np.pi / (cosmo.comoving_transverse_distance(z).value * (constants.c.value / f0))
    
def calc_2d_Pk(lc, box_dims, binning='linear', kbins=(64, 32), nu_axis=2, window=None):
    (Pk, kper_bins, kpar_bins, Nmodes) = t2c.power_spectrum_2d(lc, binning=binning, kbins=kbins, nu_axis=nu_axis,
                                                               box_dims=box_dims, return_modes=True,
                                                               window=window)
    Pk = Pk.T
    Nmodes = Nmodes.T
    return Pk, kper_bins, kpar_bins, Nmodes


def load_lc(filename, idx=0):
    # load full lightcone from a single realization
    with h5py.File(filename, 'r') as f:
        lc = f['brightness_lightcone'][idx]
    # move los axis to match t2c
    lc = np.moveaxis(lc, 0, 2)
    
    return lc