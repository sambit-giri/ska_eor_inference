import numpy as np
import tools21cm as t2c

def estimate_PdPS_4subvolumes(lc, kbins, box_length, box_length_los, verbose=False):
    lc_obs = t2c.subtract_mean_signal(lc, los_axis=2)
    ngrid = lc_obs.shape[0]
    new_ngrid = ngrid//2 #ngrid-8
    box_length_fov = box_length*(new_ngrid/ngrid)
    box_dims = [box_length_fov,box_length_fov,box_length_los]
    if verbose:
        print('Subvolume lengths', box_dims)
    
    lc_sub = lc_obs[:new_ngrid,:new_ngrid,:]
    dt_mean0 = lc_sub.mean()
    ps0, ks0 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, box_dims=box_dims)
    if verbose:
        print('Shape of Subvolume 1:', lc_sub.shape)
    lc_sub = lc_obs[:new_ngrid,-new_ngrid:,:]
    dt_mean1 = lc_sub.mean()
    ps1, ks1 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, box_dims=box_dims)
    if verbose:
        print('Shape of Subvolume 2:', lc_sub.shape)
    lc_sub = lc_obs[-new_ngrid:,:new_ngrid,:]
    dt_mean2 = lc_sub.mean()
    ps2, ks2 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, box_dims=box_dims)
    if verbose:
        print('Shape of Subvolume 3:', lc_sub.shape)
    lc_sub = lc_obs[-new_ngrid:,-new_ngrid:,:]
    dt_mean3 = lc_sub.mean()
    ps3, ks3 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, box_dims=box_dims)
    if verbose:
        print('Shape of Subvolume 4:', lc_sub.shape)
    
    iBk = (ps0*dt_mean0+ps1*dt_mean1+ps2*dt_mean2+ps3*dt_mean3)/4
    Pk_mean  = (ps0+ps1+ps2+ps3)/4
    var_mean = (dt_mean0**2+dt_mean1**2+dt_mean2**2+dt_mean3**2)/4
    return iBk/Pk_mean/var_mean, ks0

def estimate_PdPS_2subvolumes(lc, kbins, box_length, box_length_los, verbose=False):
    lc_obs = t2c.subtract_mean_signal(lc, los_axis=2)
    ngrid = lc_obs.shape[0]
    new_ngrid = ngrid//2 #ngrid-8
    box_length_fov = box_length*(new_ngrid/ngrid)
    box_dims = [box_length_fov,box_length,box_length_los]
    if verbose:
        print('Subvolume lengths', box_dims)
    
    lc_sub = lc_obs[:new_ngrid,:,:]
    dt_mean0 = lc_sub.mean()
    ps0, ks0 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, box_dims=box_dims)
    if verbose:
        print('Shape of Subvolume 1:', lc_sub.shape)
    lc_sub = lc_obs[-new_ngrid:,:,:]
    dt_mean1 = lc_sub.mean()
    ps1, ks1 = t2c.power_spectrum_1d(lc_sub, kbins=kbins, box_dims=box_dims)
    if verbose:
        print('Shape of Subvolume 2:', lc_sub.shape)
    
    iBk = (ps0*dt_mean0+ps1*dt_mean1)/2
    Pk_mean  = (ps0+ps1)/2
    var_mean = (dt_mean0**2+dt_mean1**2)/2
    return iBk/Pk_mean/var_mean, ks0

def estimate_PdPS(lc, kbins, box_length, box_length_los, verbose=False):
    out = estimate_PdPS_4subvolumes(lc, kbins, box_length, box_length_los, verbose=verbose)
    # out = estimate_PdPS_2subvolumes(lc, kbins, box_length, box_length_los, verbose=verbose)
    return out
