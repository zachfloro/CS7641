'''
ATTRIBUTIONS:
    This script makes use of numerous open source software packages that provide machine learning algorithm implementations, plotting tools, etc.
    For the decision tree, boosting, support vector machine and K-NN algorithm implementations I used the package Sklearn.  Please see the following paper for more specific information:
        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. 
        (https://scikit-learn.org/)
    For data loading and manipulation I used the Pandas package.  Please see the following paper for more specific information:
        Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232
        (https://pandas.pydata.org/)
    For data manipulation I use the Numpy package. Please see the following papers for more specifice information:
        Travis E, Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
        Stéfan van der Walt, S. Chris Colbert, and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37 (publisher link)
        (https://numpy.org/)
'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time

df = pd.read_csv('../Data/churn/Churn_Modeling.csv')