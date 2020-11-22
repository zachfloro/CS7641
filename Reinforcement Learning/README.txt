This README file explains how to download and run the code in this repository.

This project is coded entirely in python using an anaconda environment (environment.yml in the repository).

####### RUNNING THE CODE ################

1. To clone this repository open git bash and run the following command: git clone https://github.com/zachfloro/CS7641

2. Once the repository is cloned to your machine open a python prompt and navigate to the CS7641 repository and to the Reinforcement Learning sub directory on your machine.
	a.) run the following command in your python shell: conda env create -f environment.yml
	b.) run the following command in your python shell: conda activate base

3. To recreate all experiments and graphs, in your python shell run the following command: python main.py
	a.) this will create all charts used in the report
	b.) NOTE: This will take a very long time to run completely

NOTE: this code makes use of several packages (referenced below) but none of them need to be specifically modified to run this code assuming you install the conda environment in step 2


References:
Several python packages were used in this repository that I would like to acknowledge.
[1]	G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. (2016)cite arxiv:1606.01540.

[2]	Chadès, I., Chapron, G., Cros, M., Garcia, F., & Sabbadin, R. (2014). MDPtoolbox: A multi-platform toolbox to solve stochastic dynamic programming problems. Ecography, 37(9), 916-920. doi:10.1111/ecog.00888

[3] Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232
[4]  Matplotlib: A 2D graphics environment, Hunter J.D., Computing in Science & Engineering. PP 90-95, 2007.
[5] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2. (Publisher link)
