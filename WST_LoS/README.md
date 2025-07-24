This repository contains code to compute the Wavelet Scattering Transform (WST) statistic for 21cm lightcone simulations under various SKA observing configurations. The primary aim is to extract both first and second layer scattering coefficients, perform a line‑of‑sight Wavelet Transform decomposition, and evaluate Fisher‑matrix‑based constraints on model parameters.

The file compute_statistic_WST_Ian.py is where the simulations are taken, have the three cases of SKA instrument and Noise applied (AA* for 100h,AA* for 1000h, and AA4 for 1000h). The WST statistic used is the 2+1 Statistic Outlined in [Hothi et al. (2024)](https://arxiv.org/abs/2311.00036).
Here, we upload the [wavelet set](PS_2D_Window_Functions_z8_z9.npz) used in this work ${\psi_\lambda}$, where $\lambda$ is the denotes the scale of the central wavenumber of the wavelet.

For each simulated lightcone, for each case of the SKA instrument and Noise, we take a single frequency slice of the lighcone and apply the scattering transform. The First-Layer of the scattering transform is defined as:
$$
\begin{equation}
\label{eq:S1}
  S_1 (\lambda_1) = \frac{1}{\mu_1} \int |I * \psi_{\lambda_1}| (\mathbf{x}) d^2 \mathbf{x},
\end{equation}
$$
