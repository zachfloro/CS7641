'''
ATTRIBUTIONS:
    This script makes use of numerous open source software packages that provide machine learning algorithm implementations, plotting tools, etc.
    For the decision tree, boosting, support vector machine and K-NN algorithm implementations I used the package Sklearn.  Please see the following paper for more specific information:
        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. 
        (https://scikit-learn.org/)
    For data loading and manipulation I used the Pandas package.  Please see the following paper for more specific information:
        Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232
        (https://pandas.pydata.org/)
'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
