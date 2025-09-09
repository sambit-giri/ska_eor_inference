import numpy as np
import os

class  Auto_BS_Data:

    def __init__(self, path):
        self.path = path
    
    def get_k1_values(self):
        file_names = np.array(os.listdir(self.path))
        k1_values = np.sort([float(name.split('_')[-1]) for name in file_names])
        return k1_values
    
    def read_bispec_data(self):
        k1_values = self.get_k1_values()
        
        Ng=int(10) # No. of grids on the new meshgrid
        # dg = 0.5/Ng # size of each square bin

        # Declaring some arrays below
        x, y = np.linspace(0.5, 1, Ng+1), np.linspace(0.5, 1.0, Ng+1)

        ###########################################################################################
        # arrays to store the regridded bispectrum data and Ntri
        ###########################################################################################
        bispec_HI_HI_HI = np.zeros((np.size(k1_values),Ng, Ng), dtype=np.float64, order='C')
        ntri_HI_HI_HI = np.zeros((np.size(k1_values), Ng, Ng), dtype=np.float64, order='C')
        norm_bispec_HI_HI_HI = np.zeros((np.size(k1_values),Ng, Ng), dtype=np.float64, order='C')
        ###############################################################################################

        mu_bin = ((x + (x+ 0.05)) / 2)[0:-1]
        t_bin = ((y + (y+ 0.05)) / 2)[0:-1]
        
        for k in range(1, np.size(k1_values)):
            index = k
            file = '{}/bsout_k{}_{}'.format(self.path, int(index), "%.3f" % k1_values[index])
            # print(file)
            data = np.loadtxt(file)
            t = data[:,0]
            mu = data[:,1]
            HI_HI_HI = data[:,2]
            ntri_HI_HI_HI_temp = data[:,3]
            # print(data)

            for i in range(len(t)):
                ix = np.searchsorted(x, mu[i], side='right') - 1
                if ix >= Ng:
                    ix = int(ix -1)
                iy = np.searchsorted(y, t[i], side='right') - 1
                if iy >= Ng:
                    iy = int(iy -1)
                
                bispec_HI_HI_HI[k, ix, iy] += HI_HI_HI[i] * ntri_HI_HI_HI_temp[i]
                ntri_HI_HI_HI[k, ix, iy] += ntri_HI_HI_HI_temp[i]
            
            

            # normalizing the bispectrum
            for i in range(Ng):
                for j in range(Ng): 
                    norm_bispec_HI_HI_HI[k, i, j] = ((k1_values[k]**6 * t_bin[j]) / ((2 * np.pi**2)**2)) * bispec_HI_HI_HI[k, i, j]
            
        bispec_HI_HI_HI[bispec_HI_HI_HI == 0] = np.nan
        ntri_HI_HI_HI[ntri_HI_HI_HI == 0] = np.nan
        norm_bispec_HI_HI_HI[norm_bispec_HI_HI_HI == 0] = np.nan

        bispec_HI_HI_HI /= ntri_HI_HI_HI
        norm_bispec_HI_HI_HI /= ntri_HI_HI_HI
        return (k1_values, mu_bin, t_bin, norm_bispec_HI_HI_HI, ntri_HI_HI_HI, bispec_HI_HI_HI)
    
    def get_squeezed_limit(self):
        k1_values, mu_bin, t_bin, norm_bispec_HI_HI_HI, ntri_HI_HI_HI, bispec_HI_HI_HI = self.read_bispec_data()
        
        # Squeezed limit 
        squeezed_limit_HI_HI_HI = norm_bispec_HI_HI_HI[:,-1,-1]
        # sigma_sample_squeezed = squeezed_limit_HI_HI_HI / np.sqrt(ntri_HI_HI_HI[:,-1,-1])

        return (squeezed_limit_HI_HI_HI)
    
    def get_equilateral_limit(self):
        k1_values, mu_bin, t_bin, norm_bispec_HI_HI_HI, ntri_HI_HI_HI, bispec_HI_HI_HI = self.read_bispec_data()
        
        # Equilateral limit
        equilateral_limit_HI_HI_HI = norm_bispec_HI_HI_HI[:,0,-1]

        return equilateral_limit_HI_HI_HI
    



class  read_bs_data:

    def __init__(self, path):
        self.path = path
    
    def get_k1_values(self):
        file_names = np.array(os.listdir(self.path))
        k1_values = np.sort([float(name.split('_')[-1]) for name in file_names])
        return k1_values

    def get_bispec_data(self):
        k1_values = self.get_k1_values()
        bispec_data = []

        for k1 in k1_values:
            file_path = os.path.join(self.path, f"bsout_k{int(k1)}_{k1:.3f}")
            data = np.loadtxt(file_path)
            bispec_data.append(data)

        return bispec_data
    
class  read_bs_data_auto:

    def __init__(self, path):
        self.path = path
    
    def get_k1_values(self):
        file_names = np.array(os.listdir(self.path))
        k1_values = np.sort([float(name.split('_')[-1]) for name in file_names])
        return k1_values

    def get_bispec_data(self):
        k1_values = self.get_k1_values()
        bispec_data = np.array([])

        for i in range(1, np.size(k1_values)):
            file_path = os.path.join(self.path, f"bsout_k{int(i)}_{k1_values[i]:.3f}")
            data = np.loadtxt(file_path)
            bispec = data[:,2]
            bispec_data = np.concatenate((bispec_data, bispec))

        return bispec_data 