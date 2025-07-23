import numpy as np
import tools21cm as t2c
from tqdm import tqdm
try:
    import torch
    torch_available = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    torch_available = False
    torch = None

def EulerCharacteristicCurve(lc, nbins=21, speed_up='numba', verbose=False):
    if speed_up.lower()=='torch': 
        if not torch_available:
            speed_up='numba'
            print('Warning: falling to numba backend as torch is not found')
        elif device.lower()!='cuda':
            speed_up='numba'
            print('Warning: falling to numba backend as GPU is not found')
        else:
            pass
    lc_obs   = t2c.subtract_mean_signal(lc, los_axis=2)/lc.std()
    nu_bins  = np.linspace(-4,4,nbins) if isinstance(nbins,(int,float)) else nbins
    chi_vals = np.zeros_like(nu_bins)
    for i in tqdm(range(len(nu_bins)), desc="Calculating Euler Characteristics", disable=not verbose):
        thres = nu_bins[i]
        chi_vals[i] = t2c.EulerCharacteristic(lc_obs, thres=thres, neighbors=6, speed_up=speed_up, verbose=False)
    return chi_vals, nu_bins
    
