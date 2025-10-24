import os
import concurrent.futures
import numpy as np

path_data ="/scratch/leon/data/ska_ch_data/cbin/ska_AA_star_100/Lightcone_FID_400_Samples_with_noise"
#path_bs_data_out = os.path.join(path_data, "bs_data_out")
path_bs_data_out = os.path.join(path_data, "bs_data_out_new")


def process_file(index):
    input_path = os.path.join(path_data,f"realization_{int(index)}.cbin")
    output_path = os.path.join(path_bs_data_out,f"realization_{int(index)}")
    os.makedirs(output_path, exist_ok=True)
    cmd = f"./FBE_3d_mom_omp_rec input_bs_fisher_run.txt {input_path} {output_path}"
    os.system(cmd)
    print(f"{cmd}")


start_index = 0
end_index = 399
num_process = 12

index_list=np.arange(start_index,end_index,1)
print(index_list)
print(num_process)
def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers = num_process) as executor:
        executor.map(process_file,index_list)
if __name__ == '__main__':
    main()
