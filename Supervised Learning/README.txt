This README file explains how to download and run the code in this repository.

The repository containing the data and code for this project is on a public github at https://github.com/zachfloro/CS7641


This project is coded entirely in python using an anaconda environment (environment.yml in the repository).

####### RUNNING THE CODE ################

1. To clone this repository open git bash and run the following command: git clone https://github.com/zachfloro/CS7641

2. Once the repository is cloned to your machine open a python prompt and navigate to the CS7641 repository on your machine.
	a.) run the following command in your python shell: conda env create -f environment.yml
	b.) run the following command in your python shell: conda activate base

3. To run experiment 1, in your python shell run the following command: python churn.py

4. To run experiment 2, in your python shell run the following command: python chess.py

NOTE: this code makes use of several packages (referenced below) but none of them need to be specifically modified to run this code assuming you install the conda environment in step 2


References:
Several python packages were used in this repository that I would like to acknowledge.
Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

PyTorch: An Imperative Style, High-Performance Deep Learning Library. Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith (2019). Advances in Neural Information Processing Systems, 32, 8024-8035.

Matplotlib: A 2D graphics environment, Hunter J.D., Computing in Science & Engineering. PP 90-95, 2007. 

In addition data for this project was taken from Kaggle.  Special thanks to the following:
Shruti Lyyer (@shruti_lyyer). (4/3/2019). Churn modeling, version 1. Retrieved 9/5/2020 from https://www.kaggle.com/shrutimechlearn/churn-modelling

Jolly, Mitchell. (9/3/2017). Chess Game Dataset (Lichess), 1. Retrieved 9/5/2020 from https://www.kaggle.com/datasnaek/chess