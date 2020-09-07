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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
ohe = OneHotEncoder()
op_ec = X['opening_eco'].reshape(-1,1)
ohe.fit(op_ec)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)

##### Decision Tree #######

# Train a decision tree
start_time = time.time()
dt = DecisionTreeClassifier(random_state=13)
dt.fit(X_train, y_train)
end_time = time.time()
print("Decision Tree training time: " + str(end_time-start_time))
