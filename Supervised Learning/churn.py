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
    For neural networks I use the Pytorch package.  Please see the following paper for more specific information:
        PyTorch: An Imperative Style, High-Performance Deep Learning Library. Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith (2019). Advances in Neural Information Processing Systems, 32, 8024-8035.
    For graphing I used the Matplotlib package. Please see the following paper for more specific information:
        Matplotlib: A 2D graphics environment, Hunter J.D., Computing in Science & Engineering. PP 90-95, 2007. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

# Create various sizes of training set
X_train_100 = X_train.head(100)
X_train_1000 = X_train.iloc[100:].head(1000)
X_train_2500 = X_train.iloc[1100:].head(2500)
X_train_5000 = X_train.tail(5000)
y_train_100 = y_train.head(100)
y_train_1000 = y_train.iloc[100:].head(1000)
y_train_2500 = y_train.iloc[1100:].head(2500)
y_train_5000 = y_train.tail(5000)

# Do same thing for scaled versions
X_train_sc_100 = X_train_scaled.head(100)
X_train_sc_1000 = X_train_scaled.iloc[100:].head(1000)
X_train_sc_2500 = X_train_scaled.iloc[1100:].head(2500)
X_train_sc_5000 = X_train_scaled.tail(5000)

training_sets = [(X_train_100, y_train_100), (X_train_1000, y_train_1000), (X_train_2500, y_train_2500), (X_train_5000, y_train_5000), (X_train, y_train)]
training_sets_scaled = [(X_train_sc_100, y_train_100), (X_train_sc_1000, y_train_1000), (X_train_sc_2500, y_train_2500), (X_train_sc_5000, y_train_5000), (X_train_scaled, y_train)]

# Open a txt file to log data in
file = open("churn_output/churn_log.txt","w")

# Create lists to compare final test accuracy and precision for each model, plus query and train times
final_accuracy = []
final_precision = []
final_recall = []
final_train_time = []
final_query_time = []

##### Decision Tree #######

file.write("DECISION TREE RESULTS\n")

# Run cross validation on main training set to choose parameters
dt = DecisionTreeClassifier(random_state=13, criterion='entropy', class_weight='balanced')
parameters = {'max_depth':(3, 5, 10), 'min_samples_split':(10,20)} # Set parameters to be used in gridsearch
clf = GridSearchCV(dt, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')
file.write("Cross Validation Results: " + str(clf.cv_results_['mean_test_score'])+'\n')
clf_best = clf.best_estimator_


# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees with different sizes of training data
i = 1
for X, y in training_sets: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("Decision Tree training time: " + str(end_time-start_time)+'\n')
    

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for Decision Tree: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for Decision Tree: " + str(precision_score(y, y_insample))+'\n', "In sample recall for Decision Tree: " + str(recall_score(y, y_insample))+'\n', "Decision Tree insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for Decision Tree: " + str(precision_score(y_test, y_outsample))+'\n', "Out of sample recall for Decision Tree: " + str(recall_score(y_test, y_outsample))+'\n', "Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1

# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_recall.append(out_recall[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('Decision Tree In-Sample Accuracy by Training Size')
plt.savefig('churn_output/Decision Tree In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('Decision Tree Testing Accuracy by Training Size')
plt.savefig('churn_output/Decision Tree Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('Decision Tree In-Sample Precision by Training Size')
plt.savefig('churn_output/Decision Tree In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('Decision Tree Testing Precision by Training Size')
plt.savefig('churn_output/Decision Tree Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('Decision Tree In-Sample Recall by Training Size')
plt.savefig('churn_output/Decision Tree In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('Decision Tree Testing Recall by Training Size')
plt.savefig('churn_output/Decision Tree Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('Decision Tree Training Time by Training Size')
plt.savefig('churn_output/Decision Tree Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('Decision Tree In-Sample Query Time by Training Size')
plt.savefig('churn_output/Decision Tree In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('Decision Tree Testing Query Time by Training Size')
plt.savefig('churn_output/Decision Tree Testing Query Time by Training Size.png')
plt.close()
plt.figure()

##### Decision Tree w/ Boosting #######

file.write("DECISION TREE W/ BOOSTING RESULTS\n")

# Run cross validation on main training set to choose parameters
parameters = {'base_estimator__max_depth':(10,25,50), 'base_estimator__min_samples_split':(10,20,30), 'learning_rate':[0.1, 0.01, 0.001]}
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = 50, random_state=13)
clf = GridSearchCV(ada, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_
file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')
file.write("Cross Validation Results: " + str(clf.cv_results_['mean_test_score'])+'\n')

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("Boosted Decision Tree training time: " + str(end_time-start_time)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for Boosted Decision Tree: " + str(precision_score(y, y_insample))+'\n', "In sample recall for Boosted Decision Tree: " + str(recall_score(y, y_insample))+'\n', "Boosted Decision Tree insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for Boosted Decision Tree: " + str(precision_score(y_test, y_outsample))+'\n', "Out of sample recall for Boosted Decision Tree: " + str(recall_score(y_test, y_outsample))+'\n', "Boosted Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
  
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_recall.append(out_recall[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('Boosted Decision Tree In-Sample Accuracy by Training Size')
plt.savefig('churn_output/Boosted Decision Tree In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('Boosted Decision Tree Testing Accuracy by Training Size')
plt.savefig('churn_output/Boosted Decision Tree Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('Boosted Decision Tree In-Sample Precision by Training Size')
plt.savefig('churn_output/Boosted Decision Tree In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('Boosted Decision Tree Testing Precision by Training Size')
plt.savefig('churn_output/Boosted Decision Tree Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('Boosted Decision Tree In-Sample Recall by Training Size')
plt.savefig('churn_output/Boosted Decision Tree In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('Boosted Decision Tree Testing Recall by Training Size')
plt.savefig('churn_output/Boosted Decision Tree Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('Boosted Decision Tree Training Time by Training Size')
plt.savefig('churn_output/Boosted Decision Tree Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('Boosted Decision Tree In-Sample Query Time by Training Size')
plt.savefig('churn_output/Boosted Decision Tree In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('Boosted Decision Tree Testing Query Time by Training Size')
plt.savefig('churn_output/Boosted Decision Tree Testing Query Time by Training Size.png')
plt.close()
plt.figure()

# Boosting with 10 Estimators and learning rate = .8
parameters = {'learning_rate':[.8]}
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = 10, random_state=13)
clf = GridSearchCV(ada, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets: 
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    
    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    i = i+1

# Create graphs
# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('8_10 Boosted Decision Tree In-Sample Recall by Training Size')
plt.savefig('churn_output/8_10 Boosted Decision Tree In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('8_10 Boosted Decision Tree Testing Recall by Training Size')
plt.savefig('churn_output/8_10 Boosted Decision Tree Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Boosting with 100 Estimators and learning rate = .8
parameters = {'learning_rate':[.8]}
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = 100, random_state=13)
clf = GridSearchCV(ada, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets: 
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    
    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    i = i+1

# Create graphs
# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('8_100 Boosted Decision Tree In-Sample Recall by Training Size')
plt.savefig('churn_output/8_100 Boosted Decision Tree In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('8_100 Boosted Decision Tree Testing Recall by Training Size')
plt.savefig('churn_output/8_100 Boosted Decision Tree Testing Recall by Training Size.png')
plt.close()
plt.figure()


# Boosting with 10 Estimators and learning rate = .8
parameters = {'learning_rate':[.8]}
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = 10, random_state=13)
clf = GridSearchCV(ada, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets: 
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    
    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    i = i+1

# Create graphs
# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('1_10 Boosted Decision Tree In-Sample Recall by Training Size')
plt.savefig('churn_output/1_10 Boosted Decision Tree In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('1_10 Boosted Decision Tree Testing Recall by Training Size')
plt.savefig('churn_output/1_10 Boosted Decision Tree Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Boosting with 100 Estimators and learning rate = .8
parameters = {'learning_rate':[.1]}
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = 100, random_state=13)
clf = GridSearchCV(ada, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets: 
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    
    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    i = i+1

# Create graphs
# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('1_100 Boosted Decision Tree In-Sample Recall by Training Size')
plt.savefig('churn_output/1_100 Boosted Decision Tree In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('1_100 Boosted Decision Tree Testing Recall by Training Size')
plt.savefig('churn_output/1_100 Boosted Decision Tree Testing Recall by Training Size.png')
plt.close()
plt.figure()

##### K Nearest Neighbors #######

file.write("K NEAREST NEIGHBORS RESULTS\n")

# Run cross validation on main training set to choose parameters
knn = KNeighborsClassifier()
parameters = {'n_neighbors':(1,5,10,20), 'weights':('uniform','distance')}
clf = GridSearchCV(knn, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_
file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')
file.write("Cross Validation Results: " + str(clf.cv_results_['mean_test_score'])+'\n')

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets_scaled: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("KNN training time: " + str(end_time-start_time)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for KNN: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for KNN: " + str(precision_score(y, y_insample))+'\n', "In sample recall for KNN: " + str(recall_score(y, y_insample))+'\n', "KNN insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test_scaled)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for KNN: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for KNN: " + str(precision_score(y_test, y_outsample))+'\n', "Out of sample recall for KNN: " + str(recall_score(y_test, y_outsample))+'\n', "KNN out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
    
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_recall.append(out_recall[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('KNN In-Sample Accuracy by Training Size')
plt.savefig('churn_output/KNN In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('KNN Testing Accuracy by Training Size')
plt.savefig('churn_output/KNN Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('KNN In-Sample Precision by Training Size')
plt.savefig('churn_output/KNN In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('KNN Testing Precision by Training Size')
plt.savefig('churn_output/KNN Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('KNN In-Sample Recall by Training Size')
plt.savefig('churn_output/KNN In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('KNN Testing Recall by Training Size')
plt.savefig('churn_output/KNN Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('KNN Training Time by Training Size')
plt.savefig('churn_output/KNN Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('KNN In-Sample Query Time by Training Size')
plt.savefig('churn_output/KNN In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('KNN Testing Query Time by Training Size')
plt.savefig('churn_output/KNN Testing Query Time by Training Size.png')
plt.close()
plt.figure()

# Run KNN with k=1 to account for imbalanced classes
knn = KNeighborsClassifier(n_neighbors=1)
parameters = {'weights':('uniform','distance')}
clf = GridSearchCV(knn, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets_scaled: 
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test_scaled)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    i = i+1
    
# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('1NN In-Sample Recall by Training Size')
plt.savefig('churn_output/1NN In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('1NN Testing Recall by Training Size')
plt.savefig('churn_output/1NN Testing Recall by Training Size.png')
plt.close()
plt.figure()

##### Support Vector Machine #######

file.write("SUPPORT VECTOR MACHINE RESULTS\n")

# Run cross validation on main training set to choose parameters
svc = LinearSVC(random_state=13, class_weight='balanced')
parameters = {'loss':['hinge','squared_hinge'], 'tol':[1e-4, 1e-5, 0.01]}
clf = GridSearchCV(svc, parameters, 'recall', cv=5) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
clf_best = clf.best_estimator_
file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')
file.write("Cross Validation Results: " + str(clf.cv_results_['mean_test_score'])+'\n')

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets_scaled: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    clf_best.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("SVC training time: " + str(end_time-start_time)+'\n')
    file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf_best.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample))
    in_recall.append(recall_score(y, y_insample))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for SVC: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for SVC: " + str(precision_score(y, y_insample))+'\n', "In sample recall for SVC: " + str(recall_score(y, y_insample))+'\n', "SVC insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf_best.predict(X_test_scaled)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample))
    out_recall.append(recall_score(y_test, y_outsample))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for SVC: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for SVC: " + str(precision_score(y_test, y_outsample))+'\n', "Out of sample recall for SVC: " + str(recall_score(y_test, y_outsample))+'\n', "SVC out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
    
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_recall.append(out_recall[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('SVC In-Sample Accuracy by Training Size')
plt.savefig('churn_output/SVC In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('SVC Testing Accuracy by Training Size')
plt.savefig('churn_output/SVC Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('SVC In-Sample Precision by Training Size')
plt.savefig('churn_output/SVC In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('SVC Testing Precision by Training Size')
plt.savefig('churn_output/SVC Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('SVC In-Sample Recall by Training Size')
plt.savefig('churn_output/SVC In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('SVC Testing Recall by Training Size')
plt.savefig('churn_output/SVC Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('SVC Training Time by Training Size')
plt.savefig('churn_output/SVC Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('SVC In-Sample Query Time by Training Size')
plt.savefig('churn_output/SVC In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('SVC Testing Query Time by Training Size')
plt.savefig('churn_output/SVC Testing Query Time by Training Size.png')
plt.close()
plt.figure()


##### Neural Network #######

file.write("NEURAL NETWORK RESULTS\n")

# Rebalance dataset
df_train = X_train.copy()
df_train['pred'] = y_train.values
df_train_0 = df_train.loc[df_train['pred']==0]
df_train_1 = df_train.loc[df_train['pred']==1]
df_train_1_over = df_train_1.sample(df_train_0.shape[0], replace=True)
df_train = pd.concat([df_train_0, df_train_1_over],axis=0)
df_train = df_train.sample(frac=1)
X_train = df_train.iloc[:,:-1]
y_train = df_train['pred']
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

# Create various sizes of training set
X_train_100 = X_train.head(100)
X_train_1000 = X_train.iloc[100:].head(1000)
X_train_2500 = X_train.iloc[1100:].head(2500)
X_train_5000 = X_train.tail(5000)
y_train_100 = y_train.head(100)
y_train_1000 = y_train.iloc[100:].head(1000)
y_train_2500 = y_train.iloc[1100:].head(2500)
y_train_5000 = y_train.tail(5000)

# Do same thing for scaled versions
X_train_sc_100 = X_train_scaled.head(100)
X_train_sc_1000 = X_train_scaled.iloc[100:].head(1000)
X_train_sc_2500 = X_train_scaled.iloc[1100:].head(2500)
X_train_sc_5000 = X_train_scaled.tail(5000)

training_sets = [(X_train_100, y_train_100), (X_train_1000, y_train_1000), (X_train_2500, y_train_2500), (X_train_5000, y_train_5000), (X_train, y_train)]
training_sets_scaled = [(X_train_sc_100, y_train_100), (X_train_sc_1000, y_train_1000), (X_train_sc_2500, y_train_2500), (X_train_sc_5000, y_train_5000), (X_train_scaled, y_train)]

# Initialize empty lists to store data
training_loss = []
in_accuracy = []
in_precision = []
in_recall = []
out_accuracy = []
out_precision = []
out_recall = []
training_time = []
in_query_time = []
out_query_time = []

# Create class for defining MLP
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden1, self.hidden2)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden2, self.hidden3)
        self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden3, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        hidden = self.fc3(relu)
        relu = self.relu(hidden)
        output = self.fc4(relu)
        output = self.sigmoid(output)
        return output

i = 1
for X, y in training_sets_scaled: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    model = MLP(13,10,7,5)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    x_train = torch.FloatTensor(X.values)
    Y_train = torch.FloatTensor(y.values)
    x_test = torch.FloatTensor(X_test_scaled.values)
    Y_test = torch.FloatTensor(y_test.values)
    model.train()
    epoch = 10000

    for epoch in range(epoch):
        optimizer.zero_grad()
        y_insample = model(x_train)
        loss = criterion(y_insample.squeeze(), Y_train)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("NN training time: " + str(end_time-start_time)+'\n')

    model.eval()
    start_time = time.time()
    y_insample = model(x_train)
    end_time = time.time()
    in_query_time.append(end_time-start_time)
    Y_train = Y_train.detach().numpy().astype(int)
    y_insample = np.rint(y_insample.detach().numpy().flatten())
    in_accuracy.append(accuracy_score(Y_train,y_insample)) 
    in_precision.append(precision_score(Y_train,y_insample))
    in_recall.append(recall_score(Y_train,y_insample))
    file.writelines(["In sample accuracy for nn: " + str(accuracy_score(Y_train,y_insample))+'\n', "In sample precision for NN: " + str(precision_score(Y_train,y_insample))+'\n', "In sample recall for NN: " + str(recall_score(Y_train,y_insample))+'\n', "NN insample query time: " + str(end_time-start_time)+'\n'])

    start_time = time.time()
    y_outsample = model(x_test)
    end_time = time.time()
    out_query_time.append(end_time-start_time)
    after_train = criterion(y_outsample.squeeze(), Y_test)
    Y_test = Y_test.detach().numpy().astype(int)
    y_outsample = np.rint(y_outsample.detach().numpy().flatten())
    out_accuracy.append(accuracy_score(Y_test,y_outsample))
    out_precision.append(precision_score(Y_test,y_outsample))
    out_recall.append(recall_score(Y_test,y_outsample))
    file.writelines(["Out of sample accuracy for NN: " + str(accuracy_score(Y_test, y_outsample))+'\n', "Out of sample precision for NN: " + str(precision_score(Y_test, y_outsample))+'\n', "Out of sample recall for NN: " + str(recall_score(Y_test, y_outsample))+'\n', "SVC out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
    
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_recall.append(out_recall[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('NN In-Sample Accuracy by Training Size')
plt.savefig('churn_output/NN In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('NN Testing Accuracy by Training Size')
plt.savefig('churn_output/NN Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('NN In-Sample Precision by Training Size')
plt.savefig('churn_output/NN In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('NN Testing Precision by Training Size')
plt.savefig('churn_output/NN Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Recall
plt.plot(in_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Recall')
plt.title('NN In-Sample Recall by Training Size')
plt.savefig('churn_output/NN In-Sample Recall by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_recall)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Recall')
plt.title('NN Testing Recall by Training Size')
plt.savefig('churn_output/NN Testing Recall by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('NN Training Time by Training Size')
plt.savefig('churn_output/NN Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('NN In-Sample Query Time by Training Size')
plt.savefig('churn_output/NN In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,6700])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('NN Testing Query Time by Training Size')
plt.savefig('churn_output/NN Testing Query Time by Training Size.png')
plt.close()
plt.figure()


##### FINAL PLOTS #######
# Accuracy
plt.barh(np.arange(len(final_accuracy)), final_accuracy)
plt.yticks(ticks=np.arange(len(final_accuracy)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN', 'SVM', 'NN'])
plt.ylabel('Model')
plt.xlabel('Testing Accuracy')
plt.title('Testing Accuracy by Model')
plt.savefig('churn_output/Testing Accuracy by Model.png')
plt.close()
plt.figure()

# Precision
plt.barh(np.arange(len(final_precision)), final_precision)
plt.yticks(ticks=np.arange(len(final_precision)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN', 'SVM', 'NN'])
plt.ylabel('Model')
plt.xlabel('Testing Precision')
plt.title('Testing Precision by Model')
plt.savefig('churn_output/Testing Precision by Model.png')
plt.close()
plt.figure()

# Recall
plt.barh(np.arange(len(final_recall)), final_recall)
plt.yticks(ticks=np.arange(len(final_precision)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN', 'SVM', 'NN'])
plt.ylabel('Model')
plt.xlabel('Testing Recall')
plt.title('Testing Recall by Model')
plt.savefig('churn_output/Testing Recall by Model.png')
plt.close()
plt.figure()

# Training Time
plt.barh(np.arange(len(final_train_time)), final_train_time)
plt.yticks(ticks=np.arange(len(final_train_time)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN', 'SVM', 'NN'])
plt.ylabel('Model')
plt.xlabel('Training Time')
plt.title('Training Time by Model')
plt.savefig('churn_output/Training Time by Model.png')
plt.close()
plt.figure()

# Query Time
plt.barh(np.arange(len(final_query_time)), final_query_time)
plt.yticks(ticks=np.arange(len(final_query_time)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN', 'SVM', 'NN'])
plt.ylabel('Model')
plt.xlabel('Query Time')
plt.title('Query Time by Model')
plt.savefig('churn_output/Query Time by Model.png')
plt.close()
plt.figure()

#Close file
file.close()