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
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
import time

df = pd.read_csv('../Data/churn/Churn_Modeling.csv')

# Preprocessing
df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)

dummies = pd.get_dummies(df[['Geography', 'Gender']])
df.drop(columns=['Geography', 'Gender'],inplace=True)
df = df.merge(dummies, how='inner', left_index=True, right_index=True)

# Split data into train and test
y = df['Exited']
X = df.drop(columns=['Exited'])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)

# Create various sizes of training set
X_train_100 = X_train.head(100)
X_train_1000 = X_train.iloc[100:].head(1000)
X_train_2500 = X_train.iloc[1100:].head(2500)
X_train_5000 = X_train.tail(5000)
y_train_100 = y_train.head(100)
y_train_1000 = y_train.iloc[100:].head(1000)
y_train_2500 = y_train.iloc[1100:].head(2500)
y_train_5000 = y_train.tail(5000)

training_sets = [(X_train_100, y_train_100), (X_train_1000, y_train_1000), (X_train_2500, y_train_2500), (X_train_5000, y_train_5000), (X_train, y_train)]

# Open a txt file to log data in
file = open("churn_log.txt","w")

##### Decision Tree #######

file.write("DECISION TREE RESULTS\n")

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
out_accuracy = []
out_precision = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees with different sizes of training data
i = 1
for X, y in training_sets: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    dt = DecisionTreeClassifier(random_state=13)
    parameters = {'max_depth':(None, 1, 5, 10), 'min_samples_split':(2,3,4,5,6,7,8,9,10)} # Set parameters to be used in gridsearch
    clf = GridSearchCV(dt, parameters) # perform gridsearch and cross validation
    clf.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("Decision Tree training time: " + str(end_time-start_time)+'\n')
    file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for Decision Tree: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for Decision Tree: " + str(precision_score(y, y_insample))+'\n', "Decision Tree insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for Decision Tree: " + str(precision_score(y_test, y_outsample))+'\n', "Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1

# Create graphs