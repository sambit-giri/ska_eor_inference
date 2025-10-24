import os
import concurrent.futures
import numpy as np

path_data = "/scratch/leon/data/ska_ch_data/cbin"
#path_bs_data_out = os.path.join(path_data, "bs_data_out")
path_bs_data_out = os.path.join(path_data, "bs_data_out_new")

paths_params = [ "Lightcone_HII_EFF_FACTOR_400_Samples_Plus", "Lightcone_HII_EFF_FACTOR_400_Samples_Minus","Lightcone_ION_Tvir_MIN_400_Samples_Plus",
                 "Lightcone_ION_Tvir_MIN_400_Samples_Minus", "Lightcone_R_BUBBLE_MAX_400_Samples_Plus", "Lightcone_R_BUBBLE_MAX_400_Samples_Minus"]
#paths_params = ["Lightcone_FID_400_Samples"]
# for param in paths_params:
#     os.makedirs(os.path.join(path_bs_data_out, param), exist_ok=True)

def process_file(param_folder, index):
    input_path = os.path.join(path_data, param_folder, f"realization_{int(index)}.cbin")
    output_path = os.path.join(path_bs_data_out, param_folder, f"realization_{int(index)}")
    os.makedirs(output_path, exist_ok=True)
    cmd = f"./FBE_3d_mom_omp_rec input_bs_fisher_run.txt {input_path} {output_path}"
    os.system(cmd)
    # print(f"{cmd}")


start_index = 0
end_index = 399
num_process = 12

index_list=np.arange(start_index,end_index,1)

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_process) as executor:
        futures = []
        for param in paths_params:
            for idx in range(start_index, end_index):
                futures.append(executor.submit(process_file, param, idx))
        concurrent.futures.wait(futures)

if __name__ == '__main__':
    main()
