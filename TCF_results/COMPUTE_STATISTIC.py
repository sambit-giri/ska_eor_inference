
import numpy as np
import h5py
import os
from tqdm import tqdm
import glob
import re
import subprocess
import pandas as pd

#For Power Spectrum Calculations 
import tools21cm as t2c
import importlib.metadata
print(importlib.metadata.version("tools21cm"))

# for computing the TCF
from TCF_Class import *


################################ TCF Class ################################################################################################


class Compute_TCF():
    
    def __init__(self, tcf_code_dir, output_dir, nthreads=5, nbins=30, rmin=0, rmax=30):

        # files
        self.tcf_code_dir = tcf_code_dir  # directory of the files (SC.h etc) that will compute the TCF
        self.output_dir = output_dir      # directory to store output TCF data
        os.makedirs(self.output_dir, exist_ok=True) # makes output directory if it doesn't already exist

        # TCF parameters
        self.nthreads = nthreads
        self.nbins = nbins
        self.rmin = rmin
        self.rmax = rmax

    def Update_Header_File(self, input_field_filename_no_ext, L):
        """Updates SC.h with the correct parameters for a specific field."""
        
        header_path = os.path.join(self.tcf_code_dir, "SC.h")
        L = int(round(L))  # round to nearest integer
        
        with open(header_path, 'r') as f:
            content = f.read()

        content = re.sub(r'static const int nthreads = \d+;', 
                         f'static const int nthreads = {self.nthreads};', content)

        content = re.sub(r'static const string filename_box = ".*";', 
                         f'static const string filename_box = "{input_field_filename_no_ext}";', content)

        content = re.sub(r'static const int nbins = \d+;', 
                         f'static const int nbins = {self.nbins};', content)

        content = re.sub(r'static const double rmin = [\d\.]+;', 
                         f'static const double rmin = {float(self.rmin)};', content)

        content = re.sub(r'static const double rmax = [\d\.]+;', 
                         f'static const double rmax = {float(self.rmax)};', content)

        content = re.sub(r'static const double L = [\d\.]+;', 
                         f'static const double L = {float(L)};', content)

        with open(header_path, 'w') as f:
            f.write(content)

    def compute_TCF_of_single_Field(self, field_path, L):
        """
        Compute the TCF of a single input field file.
        Returns a DataFrame with r, Re_s_r, Im_s_r, N_modes.
        """
        input_field_filename_no_ext = os.path.splitext(os.path.basename(field_path))[0]
        L = int(round(L))  # round to nearest integer
        
        # Step 1: Copy the input field into the TCF code folder
        field_target_path = os.path.join(self.tcf_code_dir, os.path.basename(field_path))
        subprocess.run(["cp", field_path, field_target_path], check=True)

        # Step 2: Update SC.h
        self.Update_Header_File(input_field_filename_no_ext, L)
        print(f"Updating SC.h with filename: {input_field_filename_no_ext}")

        # Step 3: Compile and run from the TCF folder
        subprocess.run(["make"], check=True, cwd=self.tcf_code_dir)
        subprocess.run(["./SC_2d.o"], check=True, cwd=self.tcf_code_dir)

        # Step 4: Read the output
        output_filename = f"{input_field_filename_no_ext}_L{L}_spherical_correlations.txt"
        output_path = os.path.join(self.tcf_code_dir, output_filename)

        # Step 5: save output to output_dir
        final_output_path = os.path.join(self.output_dir, output_filename)
        subprocess.run(["mv", output_path, final_output_path], check=True)

        # Read the file
        data_df = pd.read_csv(final_output_path, sep=r'\s+', header=None, skiprows=2, engine='python')
        data_df.columns = ["r", "Re_s_r", "Im_s_r", "N_modes"]


        # Step 5: Clean up the copied field file
        try:
            os.remove(field_target_path)
            print(f"Deleted temporary field file: {field_target_path}")
        except Exception as e:
            print(f"Warning: could not delete file {field_target_path}: {e}")


        return data_df


################################## Add Noise to Sim ###########################################################################################



def add_noise_and_smooth_all_realisations(
    clean_h5_file,
    noisy_output_h5_file,
    obs_output_h5_file,
    noise_only_output_h5_file,
    obs_time,
    total_int_time,
    int_time,
    declination,
    subarray_type,
    verbose,
    save_uvmap,
    njobs,
    checkpoint,
    bmax_km
):
    print("Starting full SKA noise simulation for all realisations")

    # Load clean sim
    with h5py.File(clean_h5_file, 'r') as f:
        redshifts = f['redshifts'][...]
        box_length = float(f['box_length'][0])/0.6774 # Mpc
        box_dim = int(f['ngrid'][0])
        frequencies = f['frequencies'][...]
        clean_all = f['brightness_lightcone'][...]  # shape: (n_realisations, z, x, y)
        n_realisations = clean_all.shape[0]

    print(f"Loaded clean lightcone: {n_realisations} realisations of shape {clean_all.shape[1:]}")

    # Prepare output arrays
    noisy_all = np.zeros((n_realisations, box_dim, box_dim, len(redshifts)), dtype=np.float32)
    obs_all = np.zeros_like(noisy_all)
    noise_all = np.zeros_like(noisy_all)

    # Only generate and reuse UV map if not already present
    if os.path.exists(save_uvmap):
        msg = f"Reusing existing UV map: {save_uvmap}"
    else:
        msg = f"Will save UV map to: {save_uvmap}"
    
    print(msg)
    logfile_path = '/data/cluster/lcrascal/SIM_results_final/uvmaps/uvmap_log.txt'
    with open(logfile_path, "a") as log:
        log.write(msg + "\n")

    print("Looping over realisations...")

    for i in range(n_realisations):
        print(f" ➤ Processing realisation {i + 1} / {n_realisations}")

        # Move clean realisation to (x, y, z)
        clean = np.moveaxis(clean_all[i], 0, 2)  # from (z, x, y) to (x, y, z)

        # Generate a new noise lightcone for this realisation
        noise_lc = t2c.noise_lightcone(
            ncells=box_dim,
            zs=redshifts,
            obs_time=obs_time,
            total_int_time=total_int_time,
            int_time=int_time,
            declination=declination,
            subarray_type=subarray_type,
            boxsize=box_length,
            verbose=verbose,
            save_uvmap=save_uvmap,
            n_jobs=njobs,
            checkpoint=checkpoint,
        )  # shape: (box_dim, box_dim, len(z))

        #noise_lc = noise_dict['lightcone']

        noise_all[i] = noise_lc

        # Add noise
        noisy = clean + noise_lc
        noisy_all[i] = noisy

        # Smooth (subtract mean first!)
        smoothed = t2c.smooth_lightcone(
            lightcone=noise_lc + t2c.subtract_mean_signal(clean, los_axis=2),
            z_array=redshifts,
            box_size_mpc=box_length,
            max_baseline=bmax_km,
        )[0]  # (x, y, z)
        obs_all[i] = smoothed

    # === Undo the axis reorder to match original clean sim shape: (n_realisations, z, x, y)
    noisy_all = np.moveaxis(noisy_all, 3, 1)
    obs_all   = np.moveaxis(obs_all, 3, 1)
    noise_all = np.moveaxis(noise_all, 3, 1)

    def write_lightcone_to_h5(path, lightcone_data, redshifts, frequencies, box_length, box_dim, n_realisations):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, 'w') as f:
            f.create_dataset("brightness_lightcone", data=lightcone_data, compression="gzip", compression_opts=4)
            f.create_dataset("redshifts", data=redshifts)
            f.create_dataset("frequencies", data=frequencies)
            f.create_dataset("box_length", data=np.array([box_length]))
            f.create_dataset("ngrid", data=np.array([box_dim]))
            f.create_dataset("nrealisations", data=np.array([n_realisations]))


    # Save noise-only lightcones
    print(f"Saving all noise-only lightcones → {noise_only_output_h5_file}")
    write_lightcone_to_h5(path=noise_only_output_h5_file, lightcone_data=noise_all, redshifts=redshifts, 
                          frequencies=frequencies, box_length=np.array([box_length]), box_dim=np.array([box_dim]), 
                          n_realisations=np.array([n_realisations]))


    # Save noisy lightcones
    print(f"Saving noisy lightcones → {noisy_output_h5_file}")
    write_lightcone_to_h5(path=noisy_output_h5_file, lightcone_data=noisy_all, redshifts=redshifts, 
                          frequencies=frequencies, box_length=np.array([box_length]), box_dim=np.array([box_dim]), 
                          n_realisations=np.array([n_realisations]))


    # Save observed (smoothed) lightcones
    print(f"Saving smoothed lightcones → {obs_output_h5_file}")
    write_lightcone_to_h5(path=obs_output_h5_file, lightcone_data=obs_all, redshifts=redshifts, 
                          frequencies=frequencies, box_length=np.array([box_length]), box_dim=np.array([box_dim]), 
                          n_realisations=np.array([n_realisations]))


    print("All realisations processed and saved.")


################################ Extract z Slices ################################################################################################

def extract_all_z_slices_to_txt(h5_filename, output_base_dir, z_indices=None):
    """
    Extracts 2D slices at specified redshift indices (z_indices) from all realisations in a lightcone HDF5 file 
    and saves each slice as a .txt file in subdirectories of the specified base output directory.

    Parameters:
    - h5_filename (str): Path to the HDF5 file containing the lightcone data.
    - output_base_dir (str): Base directory where the output subfolders and .txt files will be saved.
    - z_indices (list, int, or None): Index or list of indices to extract. If None, extracts all slices.

    Returns:
    - A single string if one index was given, or a list of strings if multiple.
    """

    with h5py.File(h5_filename, 'r') as f:
        n_realisations = f['brightness_lightcone'].shape[0]
        frequencies = f['frequencies'][...]
        n_freq = frequencies.size

        # Normalize input
        if z_indices is None:
            z_indices = list(range(n_freq))
        elif isinstance(z_indices, int):
            z_indices = [z_indices]

        output_dirs = []

        for z_idx in z_indices:
            if z_idx < 0 or z_idx >= n_freq:
                raise ValueError(f"z_idx {z_idx} is out of bounds (0 to {n_freq - 1}).")

            # Directory for this z index
            output_dir = os.path.join(output_base_dir, f"Lightcone_zidx{z_idx}")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\nExtracting slices for z_idx={z_idx} -> {output_dir}")
                
            dataset = f['brightness_lightcone']
            if dataset.ndim == 4:  # (n_realisations, z, x, y)
                for i in range(n_realisations):
                    slice_2d = dataset[i, z_idx, :, :] # assuming shape (n_realisations, z axis, x, y axis)
                    txt_filename = os.path.join(output_dir, f"realisation_{i}.txt")
                    np.savetxt(txt_filename, slice_2d)
                
            elif dataset.ndim == 3:  # (z, x, y) → only one realisation
                slice_2d = dataset[z_idx, :, :] # assuming shape (n_realisations, z axis, x, y axis)
                txt_filename = os.path.join(output_dir, "realisation_0.txt")
                np.savetxt(txt_filename, slice_2d)
                n_realisations = 1 # for print statement 
                
            else:
                raise ValueError(f"Unexpected number of dimensions: {dataset.ndim}")

            print(f"Saved {n_realisations} slices for z_idx={z_idx}")
            output_dirs.append(output_dir)


################################ Compute TCF ################################################################################################


def compute_and_store_all_TCFs(
    tcf_code_dir,
    input_sim_h5_file,
    input_txt_folder,
    output_dir_path,
    sim_tag,
    output_dataset_name,
    nbins,
    rmin,
    rmax,
    nthreads,
    box_length,
    overwrite
):
    """
    Compute TCFs for all realisations in the input txt folder and store them in the given HDF5 file.

    Parameters:
    - tcf_code_dir: path to TCF C++ code
    - input_sim_h5_file: HDF5 file to save results into
    - input_txt_folder: folder with all .txt realisation slices
    - output_dir_path: path to folder to store output TCF data 
    - sim_tag: name for simulation identifier
    - output_dataset_name: dataset name to store (e.g. "TCF_zidx0")
    - nbins, rmin, rmax: TCF binning parameters
    - nthreads: number of threads for TCF calculation
    - box_length: simulation box size in Mpc
    - overwrite: whether to overwrite existing datasets
    """

    # Define output dir (used by Compute_TCF)
    output_dir = f"{output_dir_path}{sim_tag}_TCFs"


    # Create TCF instance
    tcf_instance = Compute_TCF(
        tcf_code_dir=tcf_code_dir,
        output_dir=output_dir,
        nthreads=nthreads,
        nbins=nbins,
        rmin=rmin,
        rmax=rmax
    )

    # Get txt files
    txt_files = sorted([f for f in os.listdir(input_txt_folder) if f.endswith(".txt")])
    n_realisations = len(txt_files)

    print(f" Found {n_realisations} realisation files in {input_txt_folder}")

    # Allocate output arrays
    tcf_Sr_vals = np.zeros((n_realisations, nbins), dtype=np.float32)
    r_values = None

    # Compute TCFs
    for idx in tqdm(range(n_realisations), desc="Computing TCFs"):
        realisation_file = txt_files[idx]
        realisation_filepath = os.path.join(input_txt_folder, realisation_file)

        # Compute TCF
        TCF_df = tcf_instance.compute_TCF_of_single_Field(realisation_filepath, box_length)
        tcf_Sr_vals[idx, :] = TCF_df["Re_s_r"].values.astype(np.float32)

        # Store r once
        if r_values is None:
            r_values = TCF_df["r"].values.astype(np.float32)

    # Save to HDF5 file
    with h5py.File(input_sim_h5_file, 'r+') as f:
        for name in [output_dataset_name, output_dataset_name + '_r']:
            if name in f:
                if overwrite:
                    print(f"Overwriting existing dataset: {name}")
                    del f[name]
                else:
                    raise ValueError(f"Dataset '{name}' already exists. Use overwrite=True to replace it.")

        f.create_dataset(output_dataset_name, data=tcf_Sr_vals, shape=tcf_Sr_vals.shape)
        f.create_dataset(output_dataset_name + '_r', data=r_values, shape=r_values.shape)

    print(f"Saved {n_realisations} TCFs to '{output_dataset_name}' in {input_sim_h5_file}'")


#############################################################################################################################################
################################ Complete Pipeline ##########################################################################################
#############################################################################################################################################

# Parameters

# home path to code
home_path = '/home/lcrascal/envs/my_env/Code/TCF/Triangle_correlations/SIMULATIONS/'
# home path to data
data_path = '/data/cluster/lcrascal/SIM_data/h5_files/sim_data_h5_files_clean/' 
txtfiles_output_path = '/data/cluster/lcrascal/SIM_results_final/AA41000h/'

# Noise Parameters (AA4, 1000hrs)
obs_time = 1000.           
total_int_time = 6.                  
int_time = 10.                        
declination = -30.0                   
subarray_type = "AA4" 
verbose = False
save_uvmap = '/data/cluster/lcrascal/SIM_results_final/uvmaps/uvmap_AA4_1000hrs.h5' 
njobs = 1
checkpoint = 16
bmax_km = 2.


# TCF global Parameters
tcf_code_dir = f"{home_path}TCF_required_files_and_functions" 
nbins = 100
rmin = 0.5
rmax = 60
nthreads = 5
box_length_Mpch = 200 #Mpc/h
box_length= 200 / 0.6774 # Mpc
overwrite = True

# SIM parameters
z_indices = 0 # which z idx to extract

# folder to all sims
sim_filepaths = glob.glob(os.path.join(data_path, "*.h5")) 

# output path of TCF files
output_dir_path = f"/data/cluster/lcrascal/SIM_results_final/AA41000h/h5_files/output_TCF_files/" 

print(subarray_type, obs_time)


# loop over all sims
for sim_path in sim_filepaths:

    print(f"%%%%%%%%%%% {sim_path} %%%%%%%%%%%")


    # for non_FID sims
    if 'FID' not in sim_path:

        ############ 1. extract z slices ###########

        # construct output_base_dir (where to store txt files)
        base = os.path.basename(sim_path)
        core = base.replace("Lightcone_", "").replace(".h5", "/")
        output_txtfiles_dir = os.path.join(txtfiles_output_path, "txt_files", core)

        # call function to extract z slices
        extract_all_z_slices_to_txt(sim_path, output_txtfiles_dir, z_indices=z_indices)
        

        ############ 2. compute TCFs of all 400 realisations ###########
        # case specific parameters
        input_txt_folder = os.path.join(output_txtfiles_dir, "Lightcone_zidx0") # for zidx0 only
        sim_tag = base.replace("Lightcone_", "").replace(".h5", "/").replace("_400_Samples","")
        output_dataset_name = "TCF_zidx0"

        # call function to compute and save TCF of all realisations and save to h5 file
        compute_and_store_all_TCFs(tcf_code_dir, sim_path, input_txt_folder, output_dir_path, sim_tag, output_dataset_name, nbins=nbins, 
                                       rmin=rmin, rmax=rmax, nthreads=nthreads, box_length=box_length, overwrite=overwrite)



    # for FID case
    if 'FID' in sim_path:

        ############ 1. add noise + smoothing ###########

        base = os.path.basename(sim_path)
        noisy_core = base.replace(".h5", "_Noisy.h5")
        obs_core = base.replace(".h5", "_Obs.h5")
        noiseonly_core = base.replace(".h5", "_Noiseonly.h5")

        # create noisy sim filenames
        noisy_sim_h5_file = os.path.join("/data/cluster/lcrascal/SIM_results_final/AA41000h/h5_files/sim_data_h5_files_noisy", noisy_core) 
        obs_sim_h5_file = os.path.join("/data/cluster/lcrascal/SIM_results_final/AA41000h/h5_files/sim_data_h5_files_noisy", obs_core) 
        noiseonly_sim_h5_file = os.path.join("/data/cluster/lcrascal/SIM_results_final/AA41000h/h5_files/sim_data_h5_files_noisy", noiseonly_core) 

        # call function to create noisy sims (sim + noise, sim + noise + smoothing, noise only lightcone) and save as new h5 files
        add_noise_and_smooth_all_realisations(sim_path, noisy_sim_h5_file, obs_sim_h5_file, noiseonly_sim_h5_file, 
                                              obs_time, total_int_time, int_time, declination, subarray_type, verbose, save_uvmap, njobs, 
                                              checkpoint, bmax_km)

        ############ 2. extract z slices ###########
        
        h5_variants = {
            "Clean": sim_path,
            "Noisy": noisy_sim_h5_file,
            "Obs": obs_sim_h5_file,
            "Noiseonly": noiseonly_sim_h5_file,
        }

        # extract z slices
        txt_dir_map = {}  # tag → txt_dir

        for tag, h5file in h5_variants.items():
            base = os.path.basename(h5file)
            core = base.replace("Lightcone_", "").replace(".h5", f"_{tag}/")
            txt_dir = os.path.join(txtfiles_output_path, "txt_files", core)
            
            extract_all_z_slices_to_txt(h5file, txt_dir, z_indices=z_indices)
            
            txt_dir_complete = os.path.join(txt_dir, "Lightcone_zidx0")
            txt_dir_map[tag] = txt_dir_complete
            

        ########### 3. compute TCFs ###########

        tcf_variants = [
            ("clean_sim", h5_variants["Clean"], txt_dir_map["Clean"]),
            ("noisy_sim", h5_variants["Noisy"], txt_dir_map["Noisy"]),
            ("obs_sim",   h5_variants["Obs"],   txt_dir_map["Obs"]),
            ("noiseonly_lc", h5_variants["Noiseonly"], txt_dir_map["Noiseonly"]),
        ]


        output_dataset_name = "TCF_zidx0"
        # loop over all types of noisy sim data
        for tag, h5file, txtdir in tcf_variants:
            compute_and_store_all_TCFs(tcf_code_dir, h5file, txtdir, output_dir_path, tag, output_dataset_name, nbins=nbins, 
                                       rmin=rmin, rmax=rmax, nthreads=nthreads, box_length=box_length, overwrite=overwrite)


