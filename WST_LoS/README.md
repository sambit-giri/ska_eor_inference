# Wavelet Scattering Transform for 21 cm Lightcone Simulations

This repository contains code to compute the Wavelet Scattering Transform (WST) statistic for 21 cm lightcone simulations under various SKA observing configurations. The primary aims are to:

1. Extract both first‑ and second‑layer scattering coefficients,  
2. Perform a line‑of‑sight (LoS) wavelet transform decomposition, and  
3. Evaluate Fisher‑matrix‑based constraints on model parameters.

The script `compute_statistic_WST_Ian.py` takes the simulated lightcones and applies three SKA cases (AA* 100 h, AA* 1000 h and AA4 1000 h) with instrumental noise. We follow the “2+1” statistic outlined in [Hothi et al. (2024)](https://arxiv.org/abs/2311.00036).

The file `PS_2D_Window_Functions_z8_z9.npz` contains the 2D wavelet set $\{\psi_\lambda\}$, where $\lambda$ denotes the central‑wavenumber scale.

We now outline the statistics used in the script `compute_statistic_WST_Ian.py`, for all the simulated lightcones (Fiducial and changes in the different parameterS) with the three SKA cases applied.
---

## 1. First‑layer Scattering Coefficients

For each frequency slice $I_z(\mathbf{x})$, the first layer is defined by

$$
\phi^{S_1}(\lambda_1, z)
= \frac{1}{\mu_1}
\int_{\mathbb{R}^2}
\bigl|\,I_z * \psi_{\lambda_1}(\mathbf{x})\bigr|\;
\mathrm{d}^2\mathbf{x},
$$

where $\mu_1$ is a normalisation constant and $*$ denotes convolution.

---

## 2. Second‑layer Scattering Coefficients

The second layer is

$$
\phi^{S_2}(\lambda_1, \lambda_2, z)
= \frac{1}{\mu_2}
\int_{\mathbb{R}^2}
\Bigl|\,
\bigl|I_z * \psi_{\lambda_1}\bigr| * \psi_{\lambda_2}(\mathbf{x})
\Bigr|\;
\mathrm{d}^2\mathbf{x},
$$

subject to the condition $\lambda_1 \le \lambda_2$, with $\mu_2$ a second normalisation constant.

---

## 3. Line‑of‑Sight Continuous Wavelet Transform

To capture the LoS evolution, we apply a continuous wavelet transform in $z$:

$$
\psi_{j_z}(t)
= \exp\!\Bigl(-\tfrac{t^2}{2^{2j_z}}\Bigr)
\;\cos\!\Bigl(\tfrac{5\,t}{2^{j_z}}\Bigr),
$$

where $j_z\in\mathbb{Z}$ is the dyadic scale parameter.

We then concatenate the first $\phi^{S_1}(z)$ and second layers $\phi^{S_2}(z)$ into
$\phi^{S}(z)$. We then perform the continuous wavelet transform and summarise with either the $\ell_1$- or $\ell_2$-norm:

$$
\bar{\phi}^{\ell_1}_{j_z}
= \bigl\|
\phi^{S}(z) * \psi_{j_z}(z)
\bigr\|_1
$$

$$
\bar{\phi}^{\ell_2}_{j_z}
= \bigl\|
\phi^{S}(z) * \psi_{j_z}(z)
\bigr\|_2^2
$$


We then concatenate these two summaries into our final statistic. To ensure that the covariances are well-conditioned we use $j_z$ = 1,2. These are then saved in HDF5 format and used in `Fisher_Tutorial_with_noise.py` to calculate the Fisher Matrices, that are saved in the `output` folder. In the `output` folder, there are files that have 'Corrected_ .txt', these are in line with the other Fisher matrices, where only the fiducial simulations are noised. The files without 'Corrected_ .txt', are where all simulations are noised. This is, in effect, a proxy for an MCMC, albeit an inaccurate proxy that assumes a Gaussian Likelihood. 

As one can see from `Fisher_Tutorial_with_noise.py`, the results for the covariance are not convergent. This will have an impact on the final results. 
