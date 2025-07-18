import numpy as np
import tools21cm as t2c
from tqdm import tqdm

def EulerCharacteristicCurve(lc, nbins=21, speed_up='numba', verbose=False):
    lc_obs   = t2c.subtract_mean_signal(lc, los_axis=2)/lc.std()
    nu_bins  = np.linspace(-4,4,nbins) if isinstance(nbins,(int,float)) else nbins
    chi_vals = np.zeros_like(nu_bins)
    for i in tqdm(range(len(nu_bins)), disable=not verbose):
        thres = nu_bins[i]
        chi_vals[i] = t2c.EulerCharacteristic(lc_obs, thres=thres, neighbors=6, speed_up=speed_up, verbose=False)
    return chi_vals, nu_bins
    
