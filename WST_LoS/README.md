# Wavelet Scattering Transform for 21cm Lightcone Simulations

This repository contains code to compute the Wavelet Scattering Transform (WST) statistic for 21 cm lightcone simulations under various SKA observing configurations. The primary aim is to

1. Extract both first‑ and second‑layer scattering coefficients,
2. Perform a line‑of‑sight (LoS) wavelet‐transform decomposition, and
3. Evaluate Fisher‐matrix‐based constraints on model parameters.

The script `compute_statistic_WST_Ian.py` takes the simulated lightcones and applies three SKA configurations (AA* 100 h, AA* 1000 h, and AA4 1000 h) including instrumental noise. We follow the “2+1” statistic outlined in [Hothi et al. (2024)](https://arxiv.org/abs/2311.00036).

The file `PS_2D_Window_Functions_z8_z9.npz` contains the 2D wavelet set \(\{\psi_\lambda\}\), where \(\lambda\) denotes the central‐wavenumber scale.

---

## 1. First‐Layer Scattering Coefficients

For each frequency slice \(I_z(\mathbf{x})\), the first layer is
\[
\phi^{S_1}(\lambda_1, z)
= \frac{1}{\mu_1}
\int_{\mathbb{R}^2}
\bigl|\,I_z * \psi_{\lambda_1}(\mathbf{x})\bigr|\,
d^2\mathbf{x},
\]
where \(\mu_1\) is a normalisation constant and “\(*\)” denotes convolution.

---

## 2. Second‐Layer Scattering Coefficients

The second layer is
\[
\phi^{S_2}(\lambda_1, \lambda_2, z)
= \frac{1}{\mu_2}
\int_{\mathbb{R}^2}
\bigl|\,
\underbrace{\bigl|I_z * \psi_{\lambda_1}\bigr| * \psi_{\lambda_2}(\mathbf{x})
}_{\text{cascaded wavelet transforms}}
\bigr|\,
d^2\mathbf{x},
\]
with the condition
\(\lambda_1 \le \lambda_2\) and \(\mu_2\) a second normalisation constant.

---

## 3. Line‑of‑Sight Continuous Wavelet Transform

To capture the LoS evolution, we apply a continuous wavelet transform in \(z\):
\[
\psi_{j_z}(t)
= \exp\!\!\Bigl(-\frac{t^2}{2^{2j_z}}\Bigr)
\;\cos\!\!\Bigl(\frac{5\,t}{2^{j_z}}\Bigr),
\]
where \(j_z\in\mathbb{Z}\) is the dyadic scale parameter.

We then concatenate the first and second layers into
\(\phi^{S}(z) = \bigl(\phi^{S_1}(z),\,\phi^{S_2}(z)\bigr)\) and compute summary statistics:

\[
\bar{\phi}^{\ell_1}_{j_z}
= \bigl\|
\phi^{S}(z) * \psi_{j_z}(z)
\bigr\|_1,
\qquad
\bar{\phi}^{\ell_2}_{j_z}
= \bigl\|
\phi^{S}(z) * \psi_{j_z}(z)
\bigr\|_2^2.
\]

---

## Files

- **compute_statistic_WST_Ian.py**  
  Main script implementing the three SKA cases and computing both layers of the WST.

- **PS_2D_Window_Functions_z8_z9.npz**  
  2D wavelet filters \(\{\psi_\lambda\}\) used for the transverse scattering transform.

---

## References

- Hothi, S. et al. (2024). *“The 2+1 Wavelet Scattering Statistic for 21 cm Cosmology.”* arXiv:2311.00036.
