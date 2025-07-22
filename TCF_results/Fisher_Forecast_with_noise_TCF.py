

import numpy as np
import h5py
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib import colors
import corner
import os
import glob
import numpy as np
import itertools


######################## Parameters ########################

######################## Global Parameters ########################


# Global params
sim_params = ['ION_Tvir_MIN','R_BUBBLE_MAX','HII_EFF_FACTOR']
Fisher_Param_labels = ['$T_{Vir}$','$R_{Max}$','$\zeta$']
nparams = len(sim_params)

# parameter values for fiducial simulation
fid_vals = [50000, 15, 30] # Tvir, Rmax, HII 

# parameter ± for derivatives
delta_params = [5000, 5, 5] # Tvir, Rmax, HII 


######################## Simulation Parameters ########################

# Read Clean FID Sim h5py file for metadata
clean_FID_sim_file = '/data/cluster/lcrascal/SIM_data/h5_files/sim_data_h5_files_clean/Lightcone_FID_400_Samples.h5'

with h5py.File(clean_FID_sim_file, 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])  # Mpc/h
    box_dim = int(f['ngrid'][0])
    n_realisations = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')


######################## Load Data ########################

######################## Data Filenames ########################

# CLEAN sims
clean_data_path = '/data/cluster/lcrascal/SIM_data/h5_files/sim_data_h5_files_clean/'
sim_filepaths_clean = glob.glob(os.path.join(clean_data_path, "*.h5")) 

# NOISY sims
sim_filepaths_noisy_AAstar100 = '/data/cluster/lcrascal/SIM_results_final/AAstar100h/h5_files/sim_data_h5_files_noisy/'
sim_filepaths_noisy_AAstar1000 = '/data/cluster/lcrascal/SIM_results_final/AAstar1000h/h5_files/sim_data_h5_files_noisy/'
sim_filepaths_noisy_AA41000 = '/data/cluster/lcrascal/SIM_results_final/AA41000h/h5_files/sim_data_h5_files_noisy/'


######################## Load Data as Dictionaries  ########################

# Initialize dictionaries
TCF_clean_data = {}
TCF_noisy_data_AAstar100 = {}
TCF_noisy_data_AAstar1000 = {}
TCF_noisy_data_AA41000 = {}

# Define config dictionary: tag → (output dictionary, file path)
noise_configs = {
    "AAstar100":  (TCF_noisy_data_AAstar100,  sim_filepaths_noisy_AAstar100),
    "AAstar1000": (TCF_noisy_data_AAstar1000, sim_filepaths_noisy_AAstar1000),
    "AA41000":    (TCF_noisy_data_AA41000,    sim_filepaths_noisy_AA41000),
}


# Tracking
keys = []
missing_count = 0

for fpath in tqdm.tqdm(sim_filepaths_clean):
    print("fpath", fpath)

    key = os.path.basename(fpath)[:-3].replace('Lightcone_', '').replace('_400_Samples', '')
    keys.append(key)

    try:
        # ---------- CLEAN SIM DATA ----------
        with h5py.File(fpath, 'r') as f:
            TCF_clean_data[key] = {
                'tcf_clean': f['TCF_zidx0'][:, :],
                'rvals': f['TCF_zidx0_r'][...],
            }

        # ---------- NOISY SIM DATA (only for FID) ----------
        if 'FID' in key:
            print("FID IN KEY")
            base = os.path.basename(fpath).replace('.h5', '')
            print("BASE", base)

            for noise_tag, (noise_dict, noise_dir) in noise_configs.items():
                noisy_key = f"{key}_{noise_tag}"
                noise_dict[noisy_key] = {}

                noisy_path     = os.path.join(noise_dir, base + '_Noisy.h5')
                obs_path       = os.path.join(noise_dir, base + '_Obs.h5')
                noiseonly_path = os.path.join(noise_dir, base + '_Noiseonly.h5')                

                # Load data from HDF5 files
                with h5py.File(noisy_path, 'r') as f_noisy:
                    noise_dict[noisy_key]['tcf_noise'] = f_noisy['TCF_zidx0'][:, :]
                    noise_dict[noisy_key]['rvals'] = f_noisy['TCF_zidx0_r'][...]

                with h5py.File(obs_path, 'r') as f_obs:
                    noise_dict[noisy_key]['tcf_obs'] = f_obs['TCF_zidx0'][:, :]

                with h5py.File(noiseonly_path, 'r') as f_noiseonly:
                    noise_dict[noisy_key]['tcf_noiseonly'] = f_noiseonly['TCF_zidx0'][:, :]

    except KeyError:
        print(f"⚠️ KeyError in: {key}")
        missing_count += 1

print(f"Loaded {len(TCF_clean_data)} clean datasets")
print(f"Loaded {len(TCF_noisy_data_AAstar100)} noisy datasets (AAstar100)")
print(f"Loaded {len(TCF_noisy_data_AAstar1000)} noisy datasets (AAstar1000)")
print(f"Loaded {len(TCF_noisy_data_AA41000)} noisy datasets (AA41000)")
print(f"{missing_count} files had missing data")


######################## Check Contents of Data Dicts ########################

def check_contents_data_dict(data_dict):
    
    for name, dat in data_dict.items():
        if isinstance(dat, dict):
            print(f"{name}:")
            for subkey, subval in dat.items():
                if hasattr(subval, "shape"):
                    print(f"    {subkey:20s} → shape {subval.shape}")
                else:
                    print(f"    {subkey:20s} → type {type(subval)}")
        else:
            print(f"{name:25s} → TCF {dat.shape},  rvals {dat['rvals'].shape}")

print("Contents of Clean SIM TCF data Dictionary:")
check_contents_data_dict(TCF_clean_data)
print("Contents of Noisy (AAstar100) SIM TCF data Dictionary:")
check_contents_data_dict(TCF_noisy_data_AAstar100)
print("Contents of Noisy (AAstar1000) SIM TCF data Dictionary:")
check_contents_data_dict(TCF_noisy_data_AAstar1000)
print("Contents of Noisy (AA41000) SIM TCF data Dictionary:")
check_contents_data_dict(TCF_noisy_data_AA41000)


######################## Re-Binning Data ########################

######################## Function to Re-Bin the Data ########################


def custom_rebin_all_tcf_data(TCF_data_dict, bin_sizes):
    """
    Rebins TCF data using custom bin sizes for each new bin.

    Parameters:
    - TCF_data_dict: dict of TCF data per sim
    - bin_sizes: list of ints, number of original bins per new bin (must sum to original bin count)

    Returns:
    - rebinned_TCF_data_dict: dict with the same structure, rebinned according to bin_sizes
    """
    rebinned_TCF_data_dict = {}

    for key, dat in TCF_data_dict.items():
        rebinned_TCF_data_dict[key] = {}

        # Rebin rvals
        if 'rvals' in dat:
            old_rvals = dat['rvals']
            n_bins = len(old_rvals)
            assert sum(bin_sizes) == n_bins, f"Sum of bin_sizes ({sum(bin_sizes)}) must equal original n_bins ({n_bins})"

            split_indices = np.cumsum(bin_sizes)[:-1]
            rvals_split = np.split(old_rvals, split_indices)
            rebinned_rvals = np.array([chunk.mean() for chunk in rvals_split])
            rebinned_TCF_data_dict[key]['rvals'] = rebinned_rvals

        # Rebin 2D TCF arrays
        for sim_name, value in dat.items():
            if sim_name == 'rvals':
                continue
            if isinstance(value, np.ndarray) and value.ndim == 2:
                tcf_array = value
                n_realisations, old_nbins = tcf_array.shape
                assert sum(bin_sizes) == old_nbins, f"Sum of bin_sizes ({sum(bin_sizes)}) must match shape {old_nbins}"
                split_indices = np.cumsum(bin_sizes)[:-1]
                tcf_split = np.split(tcf_array, split_indices, axis=1)
                rebinned_tcf = np.stack([chunk.mean(axis=1) for chunk in tcf_split], axis=1)
                rebinned_TCF_data_dict[key][sim_name] = rebinned_tcf


    return rebinned_TCF_data_dict


######################## Re-Binning the TCF Data ########################

new_bin_sizes = np.concatenate([np.full(50, 1, dtype=int),  np.full(15, 2, dtype=int),  np.full(4, 5, dtype=int)])

# Clean Data TCF
CLEAN_rebinned_TCF_data_dict = custom_rebin_all_tcf_data(TCF_clean_data, new_bin_sizes)

# AAstar100
AAstar100_rebinned_TCF_data_dict = custom_rebin_all_tcf_data(TCF_noisy_data_AAstar100, new_bin_sizes)

# AAstar1000
AAstar1000_rebinned_TCF_data_dict = custom_rebin_all_tcf_data(TCF_noisy_data_AAstar1000, new_bin_sizes)

# AA41000
AA41000_rebinned_TCF_data_dict = custom_rebin_all_tcf_data(TCF_noisy_data_AA41000, new_bin_sizes)



######################## Fisher Pipeline ########################

######################## Function to execute fisher pipeline ########################

def compute_fisher(
    tcf_clean_dict,
    tcf_noisy_dict,
    delta_params,
    sim_params,
    n_realisations,
    sim_key_noisy  
):
    """Compute Fisher matrix from TCF simulation files with separate clean and noisy data dictionaries."""

    ############ Compute derivatives from CLEAN data ############
    print("Computing Derivatives from clean simulations")
    
    rvals = tcf_clean_dict[f'{sim_params[0]}_Plus']['rvals']
    rvals_len = rvals.shape[0]
    n_realisations = tcf_clean_dict['FID']['tcf_clean'].shape[0]
    
    derivs = np.zeros((len(sim_params), n_realisations, rvals_len))
    for idx, param in enumerate(sim_params):
        plus = tcf_clean_dict[f'{param}_Plus']['tcf_clean']
        minus = tcf_clean_dict[f'{param}_Minus']['tcf_clean']
        derivs[idx] = (plus - minus) / delta_params[idx]

    ############ Whitening the Data using NOISY fiducial ############
    print(f"Whitening Data using noisy SIM: {sim_key_noisy}")
    
    TCF_noisy_sim = tcf_noisy_dict[sim_key_noisy]['tcf_obs']  # use noisy realisations for covariance
    std_noisy_sim = np.std(TCF_noisy_sim, axis=0)
    mask = std_noisy_sim > 0
    whitened_TCF_noisy_sim = TCF_noisy_sim[:, mask] / std_noisy_sim[mask]
    rvals = rvals[mask]
    derivs_white = derivs[..., mask] / std_noisy_sim[mask]

    ############ Compute Data Covariance Matrix ############
    print("Computing Data Cov Matrix")
    
    data_cov_matrix = np.cov(whitened_TCF_noisy_sim, rowvar=False)
    
    cond_num = np.log10(np.linalg.cond(data_cov_matrix))
    print(f'log10 Condition number: {cond_num:.2f}')

    ############ Compute Fisher Matrices (convergence testing) ############
    print("Convergence Testing")
    samples = np.arange(5, n_realisations + 5, 10)
    nparams = len(sim_params)
    Fisher_Matrix = np.zeros((samples.size, nparams, nparams))
    Fisher_Matrix_Inv = np.zeros((samples.size, nparams, nparams))

    inv_cov = np.linalg.inv(data_cov_matrix)
    for r, sample_size in enumerate(samples):
        deriv_sample = np.mean(derivs_white[:, :sample_size, :], axis=1)
        for i in range(nparams):
            for j in range(nparams):
                Fisher_Matrix[r, i, j] = np.dot(deriv_sample[i], inv_cov @ deriv_sample[j])
        Fisher_Matrix_Inv[r] = np.linalg.inv(Fisher_Matrix[r])

    return {
        'derivs': derivs,             # Derivatives from clean data
        'data_cov_matrix': data_cov_matrix,   # Covariance from noisy data
        'Fisher_Matrix': Fisher_Matrix,             # Fisher Matrix
        'Fisher_Matrix_Inv': Fisher_Matrix_Inv,     # Inverse Fisher Matrix
        'rvals': rvals                        # r values (masked)
    }


######################## Fisher Results for 3 Noise Configurations ########################

# AAstar100
AAstar100_FM_results_dict = compute_fisher(CLEAN_rebinned_TCF_data_dict, AAstar100_rebinned_TCF_data_dict, delta_params, 
                                               sim_params, n_realisations, sim_key_noisy='FID_AAstar100')

# AAstar1000
AAstar1000_FM_results_dict = compute_fisher(CLEAN_rebinned_TCF_data_dict, AAstar1000_rebinned_TCF_data_dict, delta_params, 
                                               sim_params, n_realisations, sim_key_noisy='FID_AAstar1000')

# AA41000
AA41000_FM_results_dict = compute_fisher(CLEAN_rebinned_TCF_data_dict, AA41000_rebinned_TCF_data_dict, delta_params, 
                                                sim_params, n_realisations, sim_key_noisy='FID_AA41000')


# Extract the final (converged) 3x3 inverse Fisher matrix = covariance matrix
res_cov_AAstar100   = AAstar100_FM_results_dict['Fisher_Matrix_Inv'][-1]
res_cov_AAstar1000  = AAstar1000_FM_results_dict['Fisher_Matrix_Inv'][-1]
res_cov_AA41000     = AA41000_FM_results_dict['Fisher_Matrix_Inv'][-1]
print(np.shape(cov_AA41000))

# Save each to a txt file
# np.savetxt("./Fisher_Results/res_cov_AAstar100.txt", res_cov_AAstar100, fmt="%.6e")
# np.savetxt("./Fisher_Results/res_cov_AAstar1000.txt", res_cov_AAstar1000, fmt="%.6e")
# np.savetxt("./Fisher_Results/res_cov_AA41000.txt", res_cov_AA41000, fmt="%.6e")


######################## Results and Plots ########################

######################## Function to Plot all Graphs ########################

def plot_fisher_results(
    results_dict,
    sim_params,
    fid_vals,
    Fisher_Param_labels,
    param_ranges=None,
    figsize_deriv=(8, 5),
    samples_title="Fisher Matrix Results"
):
    """Plot diagnostic and result figures from a Fisher matrix results dictionary."""

    ######## Unpack dictionary ########
    derivs = results_dict['derivs'] if 'derivs' in results_dict else results_dict['TCF_derivs']
    data_cov_matrix = results_dict['data_cov_matrix']
    Fisher_Matrix = results_dict['Fisher_Matrix'] if 'Fisher_Matrix' in results_dict else results_dict['TCF_Fisher']
    Fisher_Matrix_Inv = results_dict['Fisher_Matrix_Inv'] if 'Fisher_Matrix_Inv' in results_dict else results_dict['TCF_Fisher_Inv']
    rvals = results_dict['rvals']
    rvals_len = len(rvals)
    n_params = len(sim_params)

    ######## Derivative Plot ########
    colors = ['tab:red', 'tab:blue', 'tab:green']
    plt.figure(figsize=figsize_deriv)
    for ip, label in enumerate(Fisher_Param_labels):
        mean_deriv = derivs[ip].mean(axis=0)
        max_val = np.max(np.abs(mean_deriv))
        mean_norm = mean_deriv / max_val
        plt.plot(rvals, mean_norm, label=label, color=colors[ip])
    plt.xlabel(r'$r$ [Mpc]')
    plt.ylabel(r'Normalized $\partial$TCF / $\partial\theta_i$')
    plt.title(f'Mean TCF Derivatives (Normalized), N = {rvals_len}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    ######## Data Covariance Matrix Heatmap ########
    plt.figure()
    plt.pcolormesh(rvals, rvals, data_cov_matrix, cmap='coolwarm', norm=SymLogNorm(linthresh=0.1))
    plt.colorbar(label='Covariance')
    plt.title(f'Data Covariance Matrix (N={rvals_len})')
    plt.xlabel('r [Mpc]')
    plt.ylabel('r [Mpc]')
    plt.tight_layout()
    plt.show()
    plt.close()

    ######## Inverse Covariance Matrix ########
    plt.figure()
    plt.pcolormesh(rvals, rvals, np.linalg.inv(data_cov_matrix))
    plt.colorbar(label=r'$C^{-1}$')
    plt.title(f'Inverse Data Cov Matrix (N={rvals_len})')
    plt.tight_layout()
    plt.show()
    plt.close()

    ######## Stability Check ########
    I = np.log10(np.abs(np.matmul(np.linalg.inv(data_cov_matrix), data_cov_matrix)) + 1e-16)
    plt.figure()
    plt.pcolormesh(rvals, rvals, I)
    plt.colorbar(label=r'$\log_{10}(C^{-1}C)$')
    plt.title(f'Stability Check (N={rvals_len})')
    plt.tight_layout()
    plt.show()
    plt.close()

    ######## Fisher Matrix Inverse Convergence ########
    plt.figure(figsize=(12, 4))
    for i, param in enumerate(sim_params):
        y_vals = [np.diag(Fisher_Matrix_Inv[r])[i] - np.diag(Fisher_Matrix_Inv[-1])[i] for r in range(len(Fisher_Matrix_Inv))]
        max_val = np.max(np.abs(y_vals))
        y_vals_normalized = np.array(y_vals) / max_val if max_val != 0 else np.zeros_like(y_vals)
        plt.plot(range(len(y_vals)), y_vals_normalized, marker='.', label=param)
    plt.legend()
    plt.ylabel(r'$\Delta \sigma^2_{ii}$ (normalized)')
    plt.xlabel('Sample Index')
    plt.title(f'Convergence of Fisher Matrix Diagonal ({samples_title})')
    plt.tight_layout()
    plt.show()
    plt.close()

    ######## Print Final Inverse Fisher Matrix ########
    print("The Data Covariance Matrix (Inverse Fisher Matrix):")
    print(Fisher_Matrix_Inv[-1])

    ######## Corner Plot ########
    cov_final = Fisher_Matrix_Inv[-1]
    fisher_data = np.random.multivariate_normal(fid_vals, cov_final, size=100000)

    if param_ranges is None:
        param_ranges = [(None, None)] * len(fid_vals)

    corner.corner(
        fisher_data,
        labels=Fisher_Param_labels,
        plot_datapoints=False,
        levels=(0.68, 0.95),
        truths=fid_vals#,
        #range=param_ranges
    )

    ######## Compute and Return Diagnostics ########
    condition_num = np.log10(np.linalg.cond(data_cov_matrix))
    field_of_merit_num = 1.0 / np.sqrt(np.linalg.det(cov_final))

    return condition_num, field_of_merit_num


######################## Plotting all Graphs ########################

# AAstar1000
cond, fom = plot_fisher_results(
    AAstar1000_FM_results_dict,
    sim_params,
    fid_vals,
    Fisher_Param_labels,
    param_ranges=None,
    figsize_deriv=(8, 5),
    samples_title="Fisher Matrix Results AAstar1000")

print(cond, fom)


######################## Comparison of Three Corner Plots (for 3 noise configs) ########################

def overplot_fisher_corner_plots(
    fisher_results_dicts,      # list of result dicts (e.g. [dict_AA100, dict_AA1000, dict_AA41000])
    fid_vals,                  # list of fiducial values [Tvir, R_bubble, zeta]
    labels,                    # list of parameter labels (e.g. [r'$T_{\rm vir}$', r'$R_{\rm bubble}$', r'$\zeta$'])
    config_labels=None,        # optional: legend labels for each config
    colors=None,               # optional: list of colors
    n_samples=100000           # number of posterior samples to draw
):
    fig = None
    for i, results in enumerate(fisher_results_dicts):
        cov = results['Fisher_Matrix_Inv'][-1]
        samples = np.random.multivariate_normal(fid_vals, cov, size=n_samples)

        fig = corner.corner(
            samples,
            labels=labels,
            truths=fid_vals,
            fig=fig,
            color=colors[i] if colors else None,
            plot_datapoints=False,
            no_fill_contours=True,
            levels=[0.68],
            label_kwargs={"fontsize": 14},
            hist_kwargs={"density": True, "lw": 2},
            plot_density=False,
            smooth=1.0
        )

    if config_labels and len(config_labels) == len(fisher_results_dicts):
        handles = [plt.Line2D([], [], color=colors[i] if colors else "C{}".format(i), label=config_labels[i])
                   for i in range(len(config_labels))]
        fig.legend(handles=handles, loc='upper right', fontsize=12)

    plt.show()


overplot_fisher_corner_plots(
    fisher_results_dicts=[
        AAstar100_FM_results_dict,
        AAstar1000_FM_results_dict,
        AA41000_FM_results_dict
    ],
    fid_vals=fid_vals,
    labels=Fisher_Param_labels,
    config_labels=["AAstar100", "AAstar1000", "AA41000"],
    colors=["tab:red", "tab:green", "tab:orange"]
)




