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
import math
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

df = pd.read_csv('../Data/chess/games.csv')

# Feature Engineering
opening_len = [] # How long was the opening (defined as before any pieces are captured)
mid_len = [] # How long was the midgame (defined as the moves between the first captured piece (inclusive) and when a king moves for the first time and no more than 8 total pieces remain)
end_len = [] # How long was the endgame (defined as the moves after a king has finally moved and no more than 8 total pieces remain)
white_checks = [] # How many times did white put black in check
black_checks = [] # How many times did black put white in check
white_captures = [] # How many pieces did white capture
black_captures = [] # How many pieces did black capture
for index, row in df.iterrows():
    game_id = row['id']
    moves = row['moves'].split(' ')
    ids = [game_id]*len(moves)
    move_num = [x+1 for x in range(len(moves))]
    player = ['white' if x%2==1 else 'black' for x in move_num]
    move_df = pd.DataFrame(data=zip(ids,move_num,player,moves), columns=['id','move_num','player','move'])
    white_captives = 0
    black_captives = 0
    white_check_count = 0
    black_check_count = 0
    epoch = []
    game_status = 'opening'
    for idx, rw in move_df.iterrows():
        move = rw['move']
        if 'x' in move:
            if rw['player'] == 'white':
                white_captives = white_captives+1
            elif rw['player'] == 'black':
                black_captives = black_captives+1
        if '+' in move:
            if rw['player'] == 'white':
                white_check_count = white_check_count+1
            elif rw['player'] == 'black':
                black_check_count = black_check_count+1
        
        if game_status == 'opening':
            if 'K' in move:
                if white_captives+black_captives >= 24:
                    game_status = 'endgame'
                elif 'x' in move:
                    game_status = 'midgame'
            elif 'x' in move:
                game_status = 'midgame'
        elif game_status == 'midgame':
            if 'K' in move:
                if white_captives+black_captives >= 24:
                    game_status = 'endgame'
        epoch.append(game_status)
        
    opening_len.append(epoch.count('opening'))
    mid_len.append(epoch.count('midgame'))
    end_len.append(epoch.count('endgame'))
    white_checks.append(white_check_count)
    black_checks.append(black_check_count)
    white_captures.append(white_captives)
    black_captures.append(black_captives)
    
df['opening_len'] = opening_len
df['mid_len'] = mid_len
df['end_len'] = end_len
df['white_checks'] = white_checks
df['black_checks'] = black_checks
df['white_captures'] = white_captures
df['black_captures'] = black_captures

# Remove unneeded columns
df.drop(columns=['id', 'created_at', 'last_move_at', 'victory_status', 'increment_code', 'white_id', 'black_id', 'moves', 'opening_name', 'opening_ply'], inplace=True)


# Split data into train and test
y = df['winner']
X = df.drop(columns=['winner'])

# Encode string columns
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
op_ec = pd.get_dummies(X['opening_eco'])
X.drop(columns=['opening_eco'],inplace=True)
X = X.merge(op_ec, how='inner', left_index=True, right_index=True)

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
X_train_5000 = X_train.iloc[3600:].head(5000)
X_train_10000 = X_train.tail(10000)
y_train_100 = y_train[:100]
y_train_1000 = y_train[100:1100]
y_train_2500 = y_train[1100:3600]
y_train_5000 = y_train[3600:8600]
y_train_10000 = y_train[-10000:]

# Do same thing for scaled versions
X_train_sc_100 = X_train_scaled.head(100)
X_train_sc_1000 = X_train_scaled.iloc[100:].head(1000)
X_train_sc_2500 = X_train_scaled.iloc[1100:].head(2500)
X_train_sc_5000 = X_train_scaled.iloc[3600:].head(5000)
X_train_sc_10000 = X_train_scaled.tail(10000)

training_sets = [(X_train_100, y_train_100), (X_train_1000, y_train_1000), (X_train_2500, y_train_2500), (X_train_5000, y_train_5000), (X_train_10000, y_train_10000), (X_train, y_train)]
training_sets_scaled = [(X_train_sc_100, y_train_100), (X_train_sc_1000, y_train_1000), (X_train_sc_2500, y_train_2500), (X_train_sc_5000, y_train_5000), (X_train_sc_10000, y_train_10000), (X_train_scaled, y_train)]

# Create lists to compare final test accuracy and precision for each model, plus query and train times
final_accuracy = []
final_precision = []
final_train_time = []
final_query_time = []

# Open a txt file to log data in
file = open("chess_output/chess_log.txt","w")

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
    clf = GridSearchCV(dt, parameters, cv=5) # perform gridsearch and cross validation
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
    in_precision.append(precision_score(y, y_insample, average='weighted'))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for Decision Tree: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for Decision Tree: " + str(precision_score(y, y_insample, average='weighted'))+'\n', "Decision Tree insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample, average='weighted'))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for Decision Tree: " + str(precision_score(y_test, y_outsample, average='weighted'))+'\n', "Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1

# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('Decision Tree In-Sample Accuracy by Training Size')
plt.savefig('chess_output/Decision Tree In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('Decision Tree Testing Accuracy by Training Size')
plt.savefig('chess_output/Decision Tree Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('Decision Tree In-Sample Precision by Training Size')
plt.savefig('chess_output/Decision Tree In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('Decision Tree Testing Precision by Training Size')
plt.savefig('chess_output/Decision Tree Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('Decision Tree Training Time by Training Size')
plt.savefig('chess_output/Decision Tree Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('Decision Tree In-Sample Query Time by Training Size')
plt.savefig('chess_output/Decision Tree In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('Decision Tree Testing Query Time by Training Size')
plt.savefig('chess_output/Decision Tree Testing Query Time by Training Size.png')
plt.close()
plt.figure()

##### Decision Tree w/ Boosting #######

file.write("DECISION TREE W/ BOOSTING RESULTS\n")

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
out_accuracy = []
out_precision = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    parameters = {'base_estimator__max_depth':(1,5,10,25,50), 'base_estimator__min_samples_split':(5,10,15)}
    ada = AdaBoostClassifier(base_estimator = dt, n_estimators=50, random_state=13)
    clf = GridSearchCV(ada, parameters, cv=5) # perform gridsearch and cross validation
    clf.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("Boosted Decision Tree training time: " + str(end_time-start_time)+'\n')
    file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample, average='weighted'))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for Boosted Decision Tree: " + str(precision_score(y, y_insample, average='weighted'))+'\n', "Boosted Decision Tree insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf.predict(X_test)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample, average='weighted'))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for Boosted Decision Tree: " + str(precision_score(y_test, y_outsample, average='weighted'))+'\n', "Boosted Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
  
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('Boosted Decision Tree In-Sample Accuracy by Training Size')
plt.savefig('chess_output/Boosted Decision Tree In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('Boosted Decision Tree Testing Accuracy by Training Size')
plt.savefig('chess_output/Boosted Decision Tree Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('Boosted Decision Tree In-Sample Precision by Training Size')
plt.savefig('chess_output/Boosted Decision Tree In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('Boosted Decision Tree Testing Precision by Training Size')
plt.savefig('chess_output/Boosted Decision Tree Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('Boosted Decision Tree Training Time by Training Size')
plt.savefig('chess_output/Boosted Decision Tree Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('Boosted Decision Tree In-Sample Query Time by Training Size')
plt.savefig('chess_output/Boosted Decision Tree In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('Boosted Decision Tree Testing Query Time by Training Size')
plt.savefig('chess_output/Boosted Decision Tree Testing Query Time by Training Size.png')
plt.close()
plt.figure()

##### K Nearest Neighbors #######

file.write("K NEAREST NEIGHBORS RESULTS\n")

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
out_accuracy = []
out_precision = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets_scaled: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors':(1,5,10,20), 'weights':('uniform','distance')}
    clf = GridSearchCV(knn, parameters, cv=5) # perform gridsearch and cross validation
    clf.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("KNN training time: " + str(end_time-start_time)+'\n')
    file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample, average='weighted'))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for KNN: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for KNN: " + str(precision_score(y, y_insample, average='weighted'))+'\n', "KNN insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf.predict(X_test_scaled)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample, average='weighted'))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for KNN: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for KNN: " + str(precision_score(y_test, y_outsample, average='weighted'))+'\n', "KNN out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
    
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('KNN In-Sample Accuracy by Training Size')
plt.savefig('chess_output/KNN In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('KNN Testing Accuracy by Training Size')
plt.savefig('chess_output/KNN Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('KNN In-Sample Precision by Training Size')
plt.savefig('chess_output/KNN In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('KNN Testing Precision by Training Size')
plt.savefig('chess_output/KNN Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('KNN Training Time by Training Size')
plt.savefig('chess_output/KNN Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('KNN In-Sample Query Time by Training Size')
plt.savefig('chess_output/KNN In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('KNN Testing Query Time by Training Size')
plt.savefig('chess_output/KNN Testing Query Time by Training Size.png')
plt.close()
plt.figure()

##### Support Vector Machine #######

file.write("SUPPORT VECTOR MACHINE RESULTS\n")

# Initialize empty lists to store data
in_accuracy = []
in_precision = []
out_accuracy = []
out_precision = []
training_time = []
in_query_time = []
out_query_time = []

# Train decision trees w/ boosting with different sizes of training data
i = 1
for X, y in training_sets_scaled: 
    file.write('Training Set %s:\n' % (i))
    start_time = time.time()
    svc = LinearSVC(random_state=13)
    parameters = {'loss':['hinge','squared_hinge'], 'tol':[1e-4, 1e-5, 0.01]}
    clf = GridSearchCV(svc, parameters, cv=5) # perform gridsearch and cross validation
    clf.fit(X, y)
    end_time = time.time()
    training_time.append(end_time-start_time)
    file.write("SVC training time: " + str(end_time-start_time)+'\n')
    file.write("Best Classifier Chosen: " + str(clf.best_estimator_)+'\n')

    # Get predictions for in sample data
    start_time = time.time()
    y_insample = clf.predict(X)
    end_time = time.time()
    in_accuracy.append(accuracy_score(y, y_insample))
    in_precision.append(precision_score(y, y_insample, average='weighted'))
    in_query_time.append(end_time-start_time)
    file.writelines(["In sample accuracy for SVC: " + str(accuracy_score(y, y_insample))+'\n', "In sample precision for SVC: " + str(precision_score(y, y_insample, average='weighted'))+'\n', "SVC insample query time: " + str(end_time-start_time)+'\n'])

    # Get predictions for out of sample data
    start_time = time.time()
    y_outsample = clf.predict(X_test_scaled)
    end_time = time.time()
    out_accuracy.append(accuracy_score(y_test, y_outsample))
    out_precision.append(precision_score(y_test, y_outsample, average='weighted'))
    out_query_time.append(end_time-start_time)
    file.writelines(["Out of sample accuracy for SVC: " + str(accuracy_score(y_test, y_outsample))+'\n', "Out of sample precision for SVC: " + str(precision_score(y_test, y_outsample, average='weighted'))+'\n', "SVC out of sample query time: " + str(end_time-start_time)+'\n'])
    file.write("END OF ITERATION\n----------------------------------------------------------------------------------\n")
    i = i+1
    
# Append final values
final_accuracy.append(out_accuracy[-1])
final_precision.append(out_precision[-1])
final_train_time.append(training_time[-1])
final_query_time.append(out_query_time[-1])

# Create graphs
# Accuracy
plt.plot(in_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Accuracy')
plt.title('SVC In-Sample Accuracy by Training Size')
plt.savefig('chess_output/SVC In-Sample Accuracy by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_accuracy)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Accuracy')
plt.title('SVC Testing Accuracy by Training Size')
plt.savefig('chess_output/SVC Testing Accuracy by Training Size.png')
plt.close()
plt.figure()

# Precision
plt.plot(in_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Precision')
plt.title('SVC In-Sample Precision by Training Size')
plt.savefig('chess_output/SVC In-Sample Precision by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_precision)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Precision')
plt.title('SVC Testing Precision by Training Size')
plt.savefig('chess_output/SVC Testing Precision by Training Size.png')
plt.close()
plt.figure()

# Wall Time
plt.plot(training_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Training Time')
plt.title('SVC Training Time by Training Size')
plt.savefig('chess_output/SVC Training Time by Training Size.png')
plt.close()
plt.figure()

plt.plot(in_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('In-Sample Query Time')
plt.title('SVC In-Sample Query Time by Training Size')
plt.savefig('chess_output/SVC In-Sample Query by Training Size.png')
plt.close()
plt.figure()

plt.plot(out_query_time)
plt.xticks(ticks=list(range(len(training_sets))), labels=[100,1000,2500,5000,10000,13438])
plt.xlabel('Training Size')
plt.ylabel('Testing Query Time')
plt.title('SVC Testing Query Time by Training Size')
plt.savefig('chess_output/SVC Testing Query Time by Training Size.png')
plt.close()
plt.figure()

##### FINAL PLOTS #######
# Accuracy
plt.barh(np.arange(len(final_accuracy)), final_accuracy)
plt.yticks(ticks=np.arange(len(final_accuracy)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN'])
plt.ylabel('Model')
plt.xlabel('Testing Accuracy')
plt.title('Testing Accuracy by Model')
plt.savefig('chess_output/Testing Accuracy by Model.png')
plt.close()
plt.figure()

# Precision
plt.barh(np.arange(len(final_precision)), final_precision)
plt.yticks(ticks=np.arange(len(final_precision)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN'])
plt.ylabel('Model')
plt.xlabel('Testing Precision')
plt.title('Testing Precision by Model')
plt.savefig('chess_output/Testing Precision by Model.png')
plt.close()
plt.figure()

# Training Time
plt.barh(np.arange(len(final_train_time)), final_train_time)
plt.yticks(ticks=np.arange(len(final_train_time)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN'])
plt.ylabel('Model')
plt.xlabel('Training Time')
plt.title('Training Time by Model')
plt.savefig('chess_output/Training Time by Model.png')
plt.close()
plt.figure()

# Query Time
plt.barh(np.arange(len(final_query_time)), final_query_time)
plt.yticks(ticks=np.arange(len(final_query_time)), labels=['Decision Tree', 'Decision Tree w/ Boosting', 'KNN'])
plt.ylabel('Model')
plt.xlabel('Query Time')
plt.title('Query Time by Model')
plt.savefig('chess_output/Query Time by Model.png')
plt.close()
plt.figure()


#Close file
file.close()