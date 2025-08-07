# SKA EoR Inference with position-dependent power spectrum (PdPS)

This repository contains the code for the Epoch of Reionization (EoR) inference chapter of the SKA Science Working Group. **Sambit K. Giri** is the primary contributor of the code in this branch. The analysis uses the **position-dependent power spectrum (PdPS)** to derive forecasted constraints on EoR parameters. The PdPS statistic is studied in detail in [Giri et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019JCAP...02..058G/abstract).

We derive constraints on the parameters for the following cases:
1. <mark>100hrs</mark> of observations with the <mark>AA*</mark> layout,
2. <mark>1000hrs</mark> of observations with the <mark>AA*</mark> layout, and
3. <mark>1000hrs</mark> of observations with the <mark>AA4</mark> layout.

## Repository Structure

The `PdPS/` directory constains the scripts used to estimnate the corresponding summary statistic:
- `PdPS/estimator.py` contains the function used to estimate the PdPS, which relies on the [Tools21cm](https://github.com/sambit-giri/tools21cm) package
- `PdPS/compute_PdPS_*` are the scripts used to compute PdPS for the three cases.
- `PdPS/Fisher_Analysis_with_noise.ipynb` is the Jupyter notebook used to perform the Fisher analysis.
- `PdPS/Plot_Posterior.ipynb` is the Jupyter notebook showing the posterior distribution of the forecast study.
- `PdPS/SKA_chapter_statistics` is the directory containing the summaries estimated from the dataset.

The `output/` directory contains the (3,3) posterior covariance matrices for multiple scenarios. To visualize the results, use the `PdPS/Plot_Posterior.ipynb` notebook, which generates plots of the posterior distributions from the forecast.

## Usage

We recommend using a python environment using softwares, such as [venv](https://docs.python.org/3/library/venv.html) and [anaconda](https://www.anaconda.com/) to work with this package. The minimum requirements to run the scripts are listed in the `requirements.txt` file. 

## Fisher forecast

A first set of files is an example of how to obtain Fisher constraints given the set of simulations provided for the chapter. To do so, you should follow the steps:
1. Download the simulations from https://21ssd.obspm.fr, under <mark>SKA_Chapter_simulations/*h5</mark>. There is, in total, about 80GB of data. You can inspect the simulations and get more information about the files in the `load_sim` notebook and in https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.212H/abstract.
2. With the `compute_statistic.py` script, compute your statistic from the simulated lightcones (in its current version, the script computes the spherical power spectrum with [tools21cm](https://github.com/sambit-giri/tools21cm)). This will be used to compute your derivatives. The script also applies <mark>AA*</mark> layout specs (noise and beam smoothing with [tools21cm](https://github.com/sambit-giri/tools21cm)) to the simulation lightcones with fiducial parameters and compute the statistics of these observed lightcones, in order to later estimate the data covariance. The script saves the computed statistic as an additional dataset in the `hdf5` simulation files.
4. Run the `Fisher_Tutorial_with_noise` notebook to compute the Fisher covariance and obtain constraints. In its current version, the notebook computes a data covariance from sample variance between different realisations of the observed lightcone, taking into account both sample variance and noise variance. The notebook has some intermediate results, such as checks of the convergence of the Fisher covariance.
5. The end of the `Fisher_Tutorial_with_noise` notebook saves the (3,3) covariance matrix into the output folder. We expect three text files corresponding to the three cases defined above. You can share this final results by pushing the changes to your branch.
