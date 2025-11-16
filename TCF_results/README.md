# Triangle Correlation Function (TCF) Fisher Forecast Results

This folder contains my contribution to the SKA White Book EoR Inference chapter. The analysis uses the **Triangle Correlation Function (TCF)** as a non-Gaussian summary statistic to constrain astrophysical parameters from simulated 21cm observations.

---

## 📁 Folder Contents

### 🧪 Fisher Forecasts Covariance Matrix Results
- `res_cov_AAstar100.txt`  
  → 3×3 posterior covariance matrix for **100 hours** observation with **AA\*** layout.

- `res_cov_AAstar1000.txt`  
  → 3×3 posterior covariance matrix for **1000 hours** observation with **AA\*** layout.

- `res_cov_AA41000.txt`  
  → 3×3 posterior covariance matrix for **1000 hours** observation with **AA4** layout.

Each matrix is saved in plain text format, compatible with NumPy or any standard reader.

---

### 🧾 Scripts
- `COMPUTE_STATISTIC.py`  
Script for computing the **Triangle Correlation Function** from lightcone data slices. 
-	Loops through all SIMs (1 Fiducial sims + 6 parameter-perturbed variations)
-	Applies SKA-like noise and beam smoothing to the fiducial simulations (for a given noise configuration)
-	Extracts a single redshift slice of each sim for all realisations
-	Computes the 2D TCF of these slices
-	Outputs TCF results

- `Fisher_Forecast_with_noise_TCF.py`  
  Computes the **Fisher matrix** from the TCFs of clean and noisy lightcones. It:
  - Computes finite difference derivatives of the TCF wrt simulation parameters.
  - Computes the data covariance from noisy simulations (for a single noise configuration).
  - Performs whitening and computes the Fisher matrix.
  - Outputs the 3×3 posterior covariance matrix.

---

## 🛠️ Parameters and Units

- All distances are in **Mpc** (not Mpc/h). A Hubble parameter of `h = 0.6774` is assumed.
- The three parameters constrained are:
  1. `ION_Tvir_MIN` – Minimum halo virial temperature for ionizing sources  
  2. `R_BUBBLE_MAX` – Maximum bubble radius  
  3. `HII_EFF_FACTOR` – Ionizing efficiency factor

---

## 📡 Observational Effects

Noise and beam smoothing are applied to the simulations using the `tools21cm` SKA instrument model. The three observational setups match the chapter instructions:

- **AA\*** layout: 100h and 1000h
- **AA4** layout: 1000h


---

## Reproducibility 

This work was performed using:
- Python 3.11.2
- Numpy 2.2.2
- Scipy 1.15.1
- h5py 3.13.0
- Matplotlib 3.10.0
- Custom TCF computation module
- Simulations from the SKA chapter lightcones

To reproduce these results:
1. Add SKA-like noise to the fiducial lightcone simulations (`COMPUTE_STATISTIC.py`)
2. Extract 2D slices at a fixed redshift index (`COMPUTE_STATISTIC.py`)
3. Compute the TCF for each realisation (`COMPUTE_STATISTIC.py`)
4. Compute derivatives and data covariance (`Fisher_Forecast_with_noise_TCF.py`)
5. Compute the Fisher matrix and obtain posterior covariance matrices (`Fisher_Forecast_with_noise_TCF.py`)

---

## Author

**Lilian Crascall-Kennedy**  
PhD Student, IAS (Paris-Saclay)  
Contact: [lilian.crascall-kennedy@universite-paris-saclay.fr]

---
