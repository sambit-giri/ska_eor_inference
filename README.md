# SKA EoR Inference

This repository contains code relevant to the Epoch of Reionization (EoR) inference chapter for the SKA Science Working Group.

## Fisher forecast

A first set of files is an example of how to obtain Fisher constraints given the set of simulations provided for the chapter. To do so, you should follow the steps:
1. Download the simulations from https://21ssd.obspm.fr, under SKA_Chapter_simulations/. There is, in total, 175GB of data. You can inspead the simulations with the `load_sim` notebook.
2. Convert the simulations, in `npz` format, to `hdf5` format, with the `move2h5py.py` script. Make sure you update the paths to the folder where the simulations are stored and where you want to save the new ones.
3. Compute the statistic from the simulation lightcones with the `compute_statistic.py` script (in its current version, it computes the spherical power spectrum with `tools21cm`). The script saves the computed statistic as an additional dataset in the `hdf5` simulation files. New files should be about ~11GB each so half the size.
4. Run the `Fisher_forecast` notebook to compute the Fisher covariance and obtain constraints. In its current version, the notebook computes a data covariance from sample variance between different realisations of the simulation. Future versions should include the noise covariance from noise cubes computed with `tools21cm`. The notebook has some intermediate results, such as checks of the convergence of the Fisher covariance.
