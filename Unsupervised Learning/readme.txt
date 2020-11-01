This README file explains how to download and run the code in this repository.

This project is coded entirely in python using an anaconda environment (environment.yml in the repository).

####### RUNNING THE CODE ################

1. To clone this repository open git bash and run the following command: git clone https://github.com/zachfloro/CS7641

2. Once the repository is cloned to your machine open a python prompt and navigate to the CS7641 repository and to the Unsupervised Learning sub directory on your machine.
	a.) run the following command in your python shell: conda env create -f environment.yml
	b.) run the following command in your python shell: conda activate base

3. To recreate all experiments and graphs, in your python shell run the following command: python main.py
	a.) this will create all charts used in the report
	b.) NOTE: This will take a very long time to run completely

NOTE: this code makes use of several packages (referenced below) but none of them need to be specifically modified to run this code assuming you install the conda environment in step 2


References:
Several python packages were used in this repository that I would like to acknowledge.
Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

PyTorch: An Imperative Style, High-Performance Deep Learning Library. Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith (2019). Advances in Neural Information Processing Systems, 32, 8024-8035.

Matplotlib: A 2D graphics environment, Hunter J.D., Computing in Science & Engineering. PP 90-95, 2007. 

Numpy: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2. (Publisher link)

In addition data for this project was taken from Kaggle.  Special thanks to the following:
Shruti Lyyer (@shruti_lyyer). (4/3/2019). Churn modeling, version 1. Retrieved 9/5/2020 from https://www.kaggle.com/shrutimechlearn/churn-modelling

Jolly, Mitchell. (9/3/2017). Chess Game Dataset (Lichess), 1. Retrieved 9/5/2020 from https://www.kaggle.com/datasnaek/chess