# SKA EoR Inference

This repository contains code relevant to the Epoch of Reionization (EoR) inference chapter for the SKA Science Working Group.

We ask that, by July 11, you submit three (3,3) covariance matrices (of the posterior) describing the constraints on the parameters obtained with your summary statistic on a dataset obtained for the following cases:
1. <mark>100hrs</mark> of observations with the <mark>AA*</mark> layout,
2. <mark>1000hrs</mark> of observations with the <mark>AA*</mark> layout, and
3. <mark>1000hrs</mark> of observations with the <mark>AA4</mark> layout.

Note that this main repository is setup for Case 1 above. Update the total observation time (variable `obs_time` and SKA layout (variable `subarray_type`) in the codes described below for the other two cases. 

## Usage

Please create your own branch on the repo and upload your codes and result matrices (as three independent text files) to the branch by July 11, 2025. This is to ensure reproducability.

We recommend using a python environment using softwares, such as [venv](https://docs.python.org/3/library/venv.html) and [anaconda](https://www.anaconda.com/) to work with this package. The minimum requirements to run the scripts are listed in the `requirements.txt` file. 

## Fisher forecast

A first set of files is an example of how to obtain Fisher constraints given the set of simulations provided for the chapter. To do so, you should follow the steps:
1. Download the simulations from https://21ssd.obspm.fr, under <mark>SKA_Chapter_simulations/*h5</mark>. There is, in total, about 80GB of data. You can inspect the simulations and get more information about the files in the `load_sim` notebook and in https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.212H/abstract.
2. With the `compute_statistic.py` script, compute your statistic from the simulated lightcones (in its current version, the script computes the spherical power spectrum with [tools21cm](https://github.com/sambit-giri/tools21cm)). This will be used to compute your derivatives. The script also applies <mark>AA*</mark> layout specs (noise and beam smoothing with [tools21cm](https://github.com/sambit-giri/tools21cm)) to the simulation lightcones with fiducial parameters and compute the statistics of these observed lightcones, in order to later estimate the data covariance. The script saves the computed statistic as an additional dataset in the `hdf5` simulation files.
4. Run the `Fisher_Tutorial_with_noise` notebook to compute the Fisher covariance and obtain constraints. In its current version, the notebook computes a data covariance from sample variance between different realisations of the observed lightcone, taking into account both sample variance and noise variance. The notebook has some intermediate results, such as checks of the convergence of the Fisher covariance.
5. The end of the `Fisher_Tutorial_with_noise` notebook saves the (3,3) covariance matrix into the output folder. We expect three text files corresponding to the three cases defined above. You can share this final results by pushing the changes to your branch.
