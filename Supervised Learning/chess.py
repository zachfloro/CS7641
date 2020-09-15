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
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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

# Open a txt file to log data in
file = open("supervised_log.txt","w")

##### Decision Tree #######

# Train a decision tree
start_time = time.time()
dt = DecisionTreeClassifier(random_state=13)
parameters = {'max_depth':(None, 1, 5, 10), 'min_samples_split':(2,3,4,5,6,7,8,9,10)} # Set parameters to be used in gridsearch
clf = GridSearchCV(dt, parameters) # perform gridsearch and cross validation
clf.fit(X_train, y_train)
end_time = time.time()
file.write("Decision Tree training time: " + str(end_time-start_time)+'\n')
#print("Decision Tree training time: " + str(end_time-start_time))

# Get predictions for in sample data
start_time = time.time()
y_insample = clf.predict(X_train)
end_time = time.time()
file.writelines(["In sample accuracy for Decision Tree: " + str(accuracy_score(y_train, y_insample))+'\n', "Decision Tree insample query time: " + str(end_time-start_time)+'\n'])
#print("In sample accuracy for Decision Tree: " + str(accuracy_score(y_train, y_insample)))
#print("Decision Tree insample query time: " + str(end_time-start_time))

# Get predictions for out of sample data
start_time = time.time()
y_outsample = clf.predict(X_test)
end_time = time.time()
file.writelines(["Out of sample accuracy for Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
#print("Out of sample accuracy for Decision Tree: " + str(accuracy_score(y_test, y_outsample))) 
#print("Decision Tree out of sample query time: " + str(end_time-start_time))

##### Decision Tree w/ Boosting #######

# Train model using ADABoost
start_time = time.time()
parameters = {'base_estimator__max_depth':(1,5,10,25,50), 'base_estimator__min_samples_split':(5,10,15)}
ada = AdaBoostClassifier(base_estimator = dt, n_estimators=50, random_state=13)
clf = GridSearchCV(ada,parameters)
clf.fit(X_train, y_train)
end_time = time.time()
file.write("Boosted Decision Tree training time: " + str(end_time-start_time)+'\n')
#print("Boosted Decision Tree training time: " + str(end_time-start_time))

# Get predictions for in sample data
start_time = time.time()
y_insample = clf.predict(X_train)
end_time = time.time()
file.writelines(["In sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y_train, y_insample))+'\n', "Boosted Decision Tree insample query time: " + str(end_time-start_time)+'\n'])
#print("In sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y_train, y_insample)))
#print("Boosted Decision Tree insample query time: " + str(end_time-start_time))

# Get predictions for out of sample data
start_time = time.time()
y_outsample = clf.predict(X_test)
end_time = time.time()
file.writelines(["Out of sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y_test, y_outsample))+'\n', "Boosted Decision Tree out of sample query time: " + str(end_time-start_time)+'\n'])
#print("Out of sample accuracy for Boosted Decision Tree: " + str(accuracy_score(y_test, y_outsample)))
#print("Boosted Decision Tree out of sample query time: " + str(end_time-start_time))

##### K Nearest Neighbors #######

# Train a K-NN model
start_time = time.time()
knn = KNeighborsClassifier()
parameters = {'n_neighbors':(1,5,10,20), 'weights':('uniform','distance')}
clf = GridSearchCV(knn,parameters)
clf.fit(X_train, y_train)
end_time = time.time()
file.write("KNN Tree training time: " + str(end_time-start_time)+'\n')
#print("KNN training time: " + str(end_time-start_time))

# Get predictions for in sample data
start_time = time.time()
y_insample = clf.predict(X_train)
end_time = time.time()
file.writelines(["In sample accuracy for KNN: " + str(accuracy_score(y_train, y_insample))+'\n', "KNN insample query time: " + str(end_time-start_time)+'\n'])
#print("In sample accuracy for KNN: " + str(accuracy_score(y_train, y_insample)))
#print("KNN insample query time: " + str(end_time-start_time))

# Get predictions for out of sample data
start_time = time.time()
y_outsample = clf.predict(X_test)
end_time = time.time()
file.writelines(["Out of sample accuracy for KNN: " + str(accuracy_score(y_test, y_outsample))+'\n', "KNN out of sample query time: " + str(end_time-start_time)+'\n'])
#print("Out of sample accuracy for KNN: " + str(accuracy_score(y_test, y_outsample)))
#print("KNN out of sample query time: " + str(end_time-start_time))

##### Support Vector Machine #######

# Take a subsample of the training set to computational load
#X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_train, y_train, test_size=0.75, random_state=13)
## Train a SVM classifier
#start_time = time.time()
#svc = SVC(random_state=13, kernel='linear', gamma='auto', cache_size=7000)
#parameters = {'kernel':['linear'], 'gamma':['auto']}
#clf = GridSearchCV(svc, parameters)
#clf.fit(X_train_svm, y_train_svm)
#end_time = time.time()
##file.write("SVC Tree training time: " + str(end_time-start_time)+'\n')
#print("SVC training time: " + str(end_time-start_time))
#
## Get predictions for in sample data
#start_time = time.time()
#y_insample = clf.predict(X_train)
#end_time = time.time()
##file.writelines(["In sample accuracy for SVC: " + str(accuracy_score(y_train, y_insample))+'\n', "SVC insample query time: " + str(end_time-start_time)+'\n'])
#print("In sample accuracy for SVC: " + str(accuracy_score(y_train, y_insample)))
#print("SVC insample query time: " + str(end_time-start_time))
#
## Get predictions for out of sample data
#start_time = time.time()
#y_outsample = clf.predict(X_test)
#end_time = time.time()
##file.writelines(["Out of sample accuracy for SVC: " + str(accuracy_score(y_test, y_outsample))+'\n', "SVC out of sample query time: " + str(end_time-start_time)+'\n'])
#print("Out of sample accuracy for SVC: " + str(accuracy_score(y_test, y_outsample)))
#print("SVC out of sample query time: " + str(end_time-start_time))


#Close file
file.close()