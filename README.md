# SKA EoR Inference

This repository contains code relevant to the Epoch of Reionization (EoR) inference chapter for the SKA Science Working Group.

We ask that, by July 11, you submit three (3,3) covariance matrices describing the constraints obtained with your statistic on a dataset obtained for 100hrs of observations with the AA* layout, 1000hrs of observations with the AA* layout, and 1000hrs of observations with the AA4 layout.

Please create your own branch on the repo and upload your codes and result matrices (as three independent text files) to the branch by July 11. This is to ensure reproducability.

## Fisher forecast

A first set of files is an example of how to obtain Fisher constraints given the set of simulations provided for the chapter. To do so, you should follow the steps:
1. Download the simulations from https://21ssd.obspm.fr, under SKA_Chapter_simulations/*h5. There is, in total, about 80GB of data. You can inspect the simulations and get more information about the files in the `load_sim` notebook and in https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.212H/abstract.
2. With the `compute_statistic.py` script, compute your statistic from the simulated lightcones (in its current version, the script computes the spherical power spectrum with `tools21cm`). This will be used to compute your derivatives. The script also applies AA* layout specs (noise and beam smoothing with `tools21cm`) to the simulation lightcones with fiducial parameters and compute the statistics of these observed lightcones, in order to later estimate the data covariance. The script saves the computed statistic as an additional dataset in the `hdf5` simulation files.
4. Run the `Fisher_Tutorial_with_noise` notebook to compute the Fisher covariance and obtain constraints. In its current version, the notebook computes a data covariance from sample variance between different realisations of the observed lightcone, taking into account both sample variance and noise variance. The notebook has some intermediate results, such as checks of the convergence of the Fisher covariance.

Note: To include noise estimates, you need `tools21cm` with version `tools21cm>=2.3.4`.