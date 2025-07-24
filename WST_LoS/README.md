This repository contains code to compute the Wavelet Scattering Transform (WST) statistic for 21cm lightcone simulations under various SKA observing configurations. The primary aim is to extract both first and second layer scattering coefficients, perform a line‑of‑sight (LoS) Wavelet Transform decomposition, and evaluate Fisher‑matrix‑based constraints on model parameters.

The file compute_statistic_WST_Ian.py is where the simulations are taken, have the three cases of SKA instrument and Noise applied (AA* for 100h,AA* for 1000h, and AA4 for 1000h). The WST statistic used is the 2+1 Statistic Outlined in [Hothi et al. (2024)](https://arxiv.org/abs/2311.00036).
Here, we upload the [wavelet set](PS_2D_Window_Functions_z8_z9.npz) used in this work ${\psi_\lambda}$, where $\lambda$ is the denotes the scale of the central wavenumber of the wavelet.

For each simulated lightcone, for each case of the SKA instrument and Noise, we take a single frequency slice of the lighcone $I_z$ and apply the scattering transform. The First-Layer of the scattering transform is defined as:
$$
\phi^{S_1} =  (\lambda_1,z) = \frac{1}{\mu_1} \int |I_z * \psi_{\lambda_1}| (\mathbf{x}) d^2 \mathbf{x},
$$
where $\frac{1}{\mu_1}$ is a normalising factor. 
The Second-Layer is defined as:

$$ \phi^{S_2}({\lambda_1,\lambda_2,z}) = \frac{1}{\mu_2} \int||I_z * \psi_{\lambda_1}| * \psi_{\lambda_2}(\mathbf{x})d^2\mathbf{x}, $$

where $\frac{1}{\mu_2}$ is a normalising factor. There is a condition here where the scale $\lambda_2$ characterised by the second wavelet should be larger than the scale $\lambda_1$ characterised by the first wavelet, $\lambda_1 \leq \lambda_2$. 
$S_1$ and $S_2$, now contain the First and Second Layers for every frequency slice. The previously used wavelets, $psi$, where discrete wavelets but to characterise the LoS evolution of the two layers, we instead perform a wavelet transform using continous wavelets, defined as:

$$\psi_{\mathbf{j_z}}(t) = e^{-\frac{t^2}{2^{2\mathbf{j_z}}}}\cos\left(\frac{5t}{2^\mathbf{j_z}}\right)$$,

where $j_z$ is the interger scaling of the dyadic dilation. 

We conctatinate out two \phi^{S_1} and \phi^{S_2} results into a single summary \phi^{S}. We apply the continous wavelet, at a given scale $j_z$, and then summarise with either the $\ell_1$- or $\ell_2$-norm:

$$\bar{\phi}^{\ell_1}_{j_z} = || \phi^{S}(z) * \psi_{j_z}(z)||_1, ~~~~\text{and} ~~~~ \bar{\phi}^{\ell_2}_{j_z} = || \phi^{s}(z) * \psi_{j_z}(z)||_2^2.$$
