This is the branch of the inference chapter is created and edited by **Abinash Kumar Shaw**
to compute the **Multi-frequency Angular Power Spectrum (MAPS)** statistics and use it 
for further Fisher analysis to estimate the parameter error covariance.

This branch of code need following python packages
* NumPy
* SciPy
* astropy
* h5py
* corner
* tools21cm
* matplotlib

There are two main scripts in this branch. 
1) **stats.py** :  This code runs on the data set to estimate the MAPS statistics for
		   the whole data set provided. This takes roughly 18 hrs to run on a 
		   complete dataset for the first time. When repeated the run for the
		   same specifications it runs roughly in half the time.
		   This code generate the statistics and save it in the parent datafile.

2) **fisher.py** : Thus code reads the computed MAPS in the previous step and use them
		   to estimate the Fisher information matrix and hence the parameter
		   error covariance matrix. The code saves the final parameter errorcov
		   matrix to a text file and generate the corresponding corner plot. 
		   End of this code also runs the convergence test of the Fisher matrix
		   and plot the results. This code also needs to be run separately for
		   the different telescope configurations and observation hours.


