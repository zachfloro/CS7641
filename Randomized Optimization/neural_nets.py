'''
NOTE: this code makes use of several packages (referenced below) but none of them need to be specifically modified to run this code assuming you install the conda environment in step 2


References:
Several python packages were used in this repository that I would like to acknowledge.
Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232

MLROSE: Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and Search package for Python. https://github.com/gkhayes/mlrose. Accessed: 9/28/2020
MLROSE-HIIVE fork was also utilized: Rollings, A. (2020). mlrose: Machine Learning, Randomized Optimization and Search package for Python. https://pypi.org/project/mlrose-hiive/ Accessed: 9/28/2020

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

Matplotlib: A 2D graphics environment, Hunter J.D., Computing in Science & Engineering. PP 90-95, 2007. 

In addition data for this project was taken from Kaggle.  Special thanks to the following:
Shruti Lyyer (@shruti_lyyer). (4/3/2019). Churn modeling, version 1. Retrieved 9/5/2020 from https://www.kaggle.com/shrutimechlearn/churn-modelling

'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
import mlrose_hiive as mlr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

###### Neural Network Optimization ######
df = pd.read_csv('./Data/churn/Churn_Modeling.csv')

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
y_train_1000 = y_train.head(1000)
y_test_1000 = y_test.head(1000)
X_train_sc_1000 = X_train_scaled.head(1000)
X_test_sc_1000 = X_test_scaled.head(1000)

# Tune Hyperparameters for each algorithm

# Back propagation
seeds = [87323, 928056, 436302, 35012, 345571, 923279, 656314]
votes = 5*[0]
trial_times = []
trial_is_recalls = []
trial_os_recalls = []
trial_iters = []
trial_losses = []
trial_curves = []
for seed in seeds:
    best_model = None
    best_recall = -1
    fit_times = []
    fit_is_recalls = []
    fit_os_recalls = []
    fit_iters = []
    fit_losses = []
    fit_curves = []
    candidates = [mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'gradient_descent', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'gradient_descent', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'gradient_descent', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'gradient_descent', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'gradient_descent', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, early_stopping=False, curve=True, max_attempts = 100, random_state = seed)]
    for i in range(len(candidates)):
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = candidates[i]
        start_time = time.time()
        bp.fit(X_train_sc_1000, y_train_1000, init_weights=init_weights)
        end_time = time.time()
        fit_times.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_sc_1000).flatten())
        y_out = pd.Series(bp.predict(X_test_sc_1000).flatten())
        model_loss = bp.loss
        fit_losses.append(model_loss)
        fitness_curve = bp.fitness_curve
        fit_curves.append(fitness_curve)
        fit_iters.append(len(fitness_curve))
        in_sample_recall = recall_score(y_train_1000, y_in)
        fit_is_recalls.append(in_sample_recall)
        out_sample_recall = recall_score(y_test_1000, y_out)
        fit_os_recalls.append(out_sample_recall)
        if out_sample_recall > best_recall:
            best_recall = out_sample_recall
            best_model = i
    trial_times.append(fit_times)
    trial_is_recalls.append(fit_is_recalls)
    trial_os_recalls.append(fit_os_recalls)
    trial_iters.append(fit_iters)
    trial_losses.append(fit_losses)
    trial_curves.append(fit_curves)
    votes = np.add(votes, fit_os_recalls)

# Identify the best candidate model
final_model = candidates[np.argmax(votes)]
bp_lr = final_model.learning_rate

# Graph the average recalls for each of the models
candidate_names = ['LR=0.001', 'LR=0.0001', 'LR=0.00001', 'LR=0.000001', 'LR=0.0000001']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
trial_os_recalls = np.array(trial_os_recalls)
os_recall_means = np.mean(trial_os_recalls, axis=0)
trial_is_recalls = np.array(trial_is_recalls)
is_recall_means = np.mean(trial_os_recalls, axis=0)
fig, (ax1, ax2) = plt.subplots(2)
plt.setp(ax1, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax1, yticklabels=candidate_names)
plt.setp(ax2, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax2, yticklabels=candidate_names)
ax1.barh(np.arange(len(os_recall_means)), os_recall_means, color=colors)
ax1.set_title('Average Recall Score for Test Data')
ax2.barh(np.arange(len(is_recall_means)),is_recall_means, color=colors)
ax2.set_title('Average Recall Score for In-Sample Data')
fig.suptitle('Learning Rate Tuning Results for Gradient Descent')
plt.tight_layout()
fig.savefig('Output/NN_GD_LR_Tuning.png', bbox_inches='tight')


# Random Hill Climbing
seeds = [87323, 928056, 436302, 35012, 345571, 923279, 656314]
votes = 15*[0]
trial_times = []
trial_is_recalls = []
trial_os_recalls = []
trial_iters = []
trial_losses = []
trial_curves = []
for seed in seeds:
    best_model = None
    best_recall = -1
    fit_times = []
    fit_is_recalls = []
    fit_os_recalls = []
    fit_iters = []
    fit_losses = []
    fit_curves = []
    candidates = [mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, restarts = 5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, restarts = 5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, restarts = 5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, restarts = 5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, restarts = 5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed), \
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, restarts = 15, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, restarts = 15, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, restarts = 15, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, restarts = 15, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, restarts = 15, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, restarts = 25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, restarts = 25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, restarts = 25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, restarts = 25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, restarts = 25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed)]
    for i in range(len(candidates)):
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = candidates[i]
        start_time = time.time()
        bp.fit(X_train_sc_1000, y_train_1000, init_weights=init_weights)
        end_time = time.time()
        fit_times.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_sc_1000).flatten())
        y_out = pd.Series(bp.predict(X_test_sc_1000).flatten())
        model_loss = bp.loss
        fit_losses.append(model_loss)
        fitness_curve = bp.fitness_curve
        fit_curves.append(fitness_curve)
        fit_iters.append(len(fitness_curve))
        in_sample_recall = recall_score(y_train_1000, y_in)
        fit_is_recalls.append(in_sample_recall)
        out_sample_recall = recall_score(y_test_1000, y_out)
        fit_os_recalls.append(out_sample_recall)
        if out_sample_recall > best_recall:
            best_recall = out_sample_recall
            best_model = i
    trial_times.append(fit_times)
    trial_is_recalls.append(fit_is_recalls)
    trial_os_recalls.append(fit_os_recalls)
    trial_iters.append(fit_iters)
    trial_losses.append(fit_losses)
    trial_curves.append(fit_curves)
    votes = np.add(votes, fit_os_recalls)

# Identify the best candidate model
final_model = candidates[np.argmax(votes)]
rhc_lr = final_model.learning_rate
rhc_restarts = final_model.restarts

# Graph the average recalls for each of the models
candidate_names = ['0.001, 5', '0.0001, 5', '0.00001, 5', '0.000001, 5', '0.0000001, 5', '0.001, 15', '0.0001, 15', '0.00001, 15', '0.000001, 15', '0.0000001, 15', '0.001, 25', '0.0001, 25', '0.00001, 25', '0.000001, 25', '0.0000001, 25']
colors = 3*['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
trial_os_recalls = np.array(trial_os_recalls)
os_recall_means = np.mean(trial_os_recalls, axis=0)
trial_is_recalls = np.array(trial_is_recalls)
is_recall_means = np.mean(trial_os_recalls, axis=0)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
plt.setp(ax1, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax1, yticklabels=candidate_names)
plt.setp(ax2, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax2, yticklabels=candidate_names)
ax1.barh(np.arange(len(os_recall_means)), os_recall_means, color=colors)
ax1.set_title('Average Recall Score for Test Data')
ax2.barh(np.arange(len(is_recall_means)),is_recall_means, color=colors)
ax2.set_title('Average Recall Score for In-Sample Data')
fig.suptitle('Learning Rate Tuning Results for Random Hill Climbing')
plt.tight_layout()
fig.savefig('Output/NN_RHC_LR_Tuning.png', bbox_inches='tight')


# Simulated Annealing
seeds = [87323, 928056, 436302, 35012, 345571, 923279, 656314]
votes = 15*[0]
trial_times = []
trial_is_recalls = []
trial_os_recalls = []
trial_iters = []
trial_losses = []
trial_curves = []
for seed in seeds:
    best_model = None
    best_recall = -1
    fit_times = []
    fit_is_recalls = []
    fit_os_recalls = []
    fit_iters = []
    fit_losses = []
    fit_curves = []
    candidates = [mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, schedule=mlr.GeomDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, schedule=mlr.GeomDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, schedule=mlr.GeomDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, schedule=mlr.GeomDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, schedule=mlr.GeomDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed), \
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, schedule=mlr.ArithDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, schedule=mlr.ArithDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, schedule=mlr.ArithDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, schedule=mlr.ArithDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, schedule=mlr.ArithDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, schedule=mlr.ExpDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, schedule=mlr.ExpDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, schedule=mlr.ExpDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, schedule=mlr.ExpDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, schedule=mlr.ExpDecay(), early_stopping=False, curve=True, max_attempts = 100, random_state = seed)]
    for i in range(len(candidates)):
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = candidates[i]
        start_time = time.time()
        bp.fit(X_train_sc_1000, y_train_1000, init_weights=init_weights)
        end_time = time.time()
        fit_times.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_sc_1000).flatten())
        y_out = pd.Series(bp.predict(X_test_sc_1000).flatten())
        model_loss = bp.loss
        fit_losses.append(model_loss)
        fitness_curve = bp.fitness_curve
        fit_curves.append(fitness_curve)
        fit_iters.append(len(fitness_curve))
        in_sample_recall = recall_score(y_train_1000, y_in)
        fit_is_recalls.append(in_sample_recall)
        out_sample_recall = recall_score(y_test_1000, y_out)
        fit_os_recalls.append(out_sample_recall)
        if out_sample_recall > best_recall:
            best_recall = out_sample_recall
            best_model = i
    trial_times.append(fit_times)
    trial_is_recalls.append(fit_is_recalls)
    trial_os_recalls.append(fit_os_recalls)
    trial_iters.append(fit_iters)
    trial_losses.append(fit_losses)
    trial_curves.append(fit_curves)
    votes = np.add(votes, fit_os_recalls)

# Identify the best candidate model
final_model = candidates[np.argmax(votes)]
sa_lr = final_model.learning_rate
sa_sched = final_model.schedule

# Graph the average recalls for each of the models
candidate_names = ['0.001, Geom', '0.0001, Geom', '0.00001, Geom', '0.000001, Geom', '0.0000001, Geom', '0.001, Arith', '0.0001, Arith', '0.00001, Arith', '0.000001, Arith', '0.0000001, Arith', '0.001, Exp', '0.0001, Exp', '0.00001, Exp', '0.000001, Exp', '0.0000001, Exp']
colors = 3*['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
trial_os_recalls = np.array(trial_os_recalls)
os_recall_means = np.mean(trial_os_recalls, axis=0)
trial_is_recalls = np.array(trial_is_recalls)
is_recall_means = np.mean(trial_os_recalls, axis=0)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
plt.setp(ax1, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax1, yticklabels=candidate_names)
plt.setp(ax2, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax2, yticklabels=candidate_names)
ax1.barh(np.arange(len(os_recall_means)), os_recall_means, color=colors)
ax1.set_title('Average Recall Score for Test Data')
ax2.barh(np.arange(len(is_recall_means)),is_recall_means, color=colors)
ax2.set_title('Average Recall Score for In-Sample Data')
fig.suptitle('Learning Rate Tuning Results for Simulated Annealing')
plt.tight_layout()
fig.savefig('Output/NN_SA_LR_Tuning.png', bbox_inches='tight')


# Genetic Algorithm
seeds = [87323, 928056, 436302, 35012, 345571, 923279, 656314]
votes = 15*[0]
trial_times = []
trial_is_recalls = []
trial_os_recalls = []
trial_iters = []
trial_losses = []
trial_curves = []
for seed in seeds:
    best_model = None
    best_recall = -1
    fit_times = []
    fit_is_recalls = []
    fit_os_recalls = []
    fit_iters = []
    fit_losses = []
    fit_curves = []
    candidates = [mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, mutation_prob=0.1, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, mutation_prob=0.1, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, mutation_prob=0.1, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, mutation_prob=0.1, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, mutation_prob=0.1, early_stopping=False, curve=True, max_attempts = 100, random_state = seed), \
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, mutation_prob=0.25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, mutation_prob=0.25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, mutation_prob=0.25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, mutation_prob=0.25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, mutation_prob=0.25, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.001, mutation_prob=0.5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, mutation_prob=0.5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001, mutation_prob=0.5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.000001, mutation_prob=0.5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed),\
                  mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0000001, mutation_prob=0.5, early_stopping=False, curve=True, max_attempts = 100, random_state = seed)]
    for i in range(len(candidates)):
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = candidates[i]
        start_time = time.time()
        bp.fit(X_train_sc_1000, y_train_1000, init_weights=init_weights)
        end_time = time.time()
        fit_times.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_sc_1000).flatten())
        y_out = pd.Series(bp.predict(X_test_sc_1000).flatten())
        model_loss = bp.loss
        fit_losses.append(model_loss)
        fitness_curve = bp.fitness_curve
        fit_curves.append(fitness_curve)
        fit_iters.append(len(fitness_curve))
        in_sample_recall = recall_score(y_train_1000, y_in)
        fit_is_recalls.append(in_sample_recall)
        out_sample_recall = recall_score(y_test_1000, y_out)
        fit_os_recalls.append(out_sample_recall)
        if out_sample_recall > best_recall:
            best_recall = out_sample_recall
            best_model = i
    trial_times.append(fit_times)
    trial_is_recalls.append(fit_is_recalls)
    trial_os_recalls.append(fit_os_recalls)
    trial_iters.append(fit_iters)
    trial_losses.append(fit_losses)
    trial_curves.append(fit_curves)
    votes = np.add(votes, fit_os_recalls)

# Identify the best candidate model
final_model = candidates[np.argmax(votes)]
ga_lr = final_model.learning_rate
ga_prob = final_model.mutation_prob

# Graph the average recalls for each of the models
candidate_names = ['0.001, 0.1', '0.0001, 0.1', '0.00001, 0.1', '0.000001, 0.1', '0.0000001, 0.1', '0.001, 0.25', '0.0001, 0.25', '0.00001, 0.25', '0.000001, 0.25', '0.0000001, 0.25', '0.001, 0.5', '0.0001, 0.5', '0.00001, 0.5', '0.000001, 0.5', '0.0000001, 0.5']
colors = 3*['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
trial_os_recalls = np.array(trial_os_recalls)
os_recall_means = np.mean(trial_os_recalls, axis=0)
trial_is_recalls = np.array(trial_is_recalls)
is_recall_means = np.mean(trial_os_recalls, axis=0)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
plt.setp(ax1, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax1, yticklabels=candidate_names)
plt.setp(ax2, yticks=[x for x in range(len(candidate_names))])
plt.setp(ax2, yticklabels=candidate_names)
ax1.barh(np.arange(len(os_recall_means)), os_recall_means, color=colors)
ax1.set_title('Average Recall Score for Test Data')
ax2.barh(np.arange(len(is_recall_means)),is_recall_means, color=colors)
ax2.set_title('Average Recall Score for In-Sample Data')
fig.suptitle('Learning Rate Tuning Results for Genetic Algorithm')
plt.tight_layout()
fig.savefig('Output/NN_GA_LR_Tuning.png', bbox_inches='tight')

# TODO trials for each algorithm limiting the number of allowed iterations and compare recall score
# TODO show the loss curve for each algorithm by iteration ()

# Performance Per Iteration
iterations = [5, 50, 100, 500, 1000, 2500]
algo_losses = []
algo_times = []
algo_os_recall = []
algo_is_recall = []

# Back propagation
bp_lr = 0.0001
is_recall = []
os_recall = []
training_times = []
training_losses = []
for i in iterations:
    temp_is = []
    temp_os = []
    temp_time = []
    temp_loss = []
    for seed in seeds:
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'gradient_descent', max_iters=i, bias=True, is_classifier=True, learning_rate=bp_lr, early_stopping=False, curve=True, random_state = seed)
        start_time = time.time()
        bp.fit(X_train_scaled, y_train, init_weights=init_weights)
        end_time = time.time()
        temp_time.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_scaled).flatten())
        y_out = pd.Series(bp.predict(X_test_scaled).flatten())
        model_loss = bp.loss
        temp_loss.append(model_loss)
        in_sample_recall = recall_score(y_train, y_in)
        temp_is.append(in_sample_recall)
        out_sample_recall = recall_score(y_test, y_out)
        temp_os.append(out_sample_recall)
    is_recall.append(temp_is)
    os_recall.append(temp_os)
    training_times.append(temp_time)
    training_losses.append(temp_loss)
# Conver(t data into numpy arrays of averages per iterations
is_recall = np.mean(np.array(is_recall), axis=1)
os_recall = np.mean(np.array(os_recall), axis=1)
training_times = np.mean(np.array(training_times), axis=1)
training_losses = np.mean(np.array(training_losses), axis=1)
# Add to overall lists
algo_losses.append(training_losses)
algo_times.append(training_times)
algo_os_recall.append(os_recall)
algo_is_recall.append(is_recall)

# Random Hill Climbing
rhc_lr = 0.000001
rhc_restarts = 5
is_recall = []
os_recall = []
training_times = []
training_losses = []
for i in iterations:
    temp_is = []
    temp_os = []
    temp_time = []
    temp_loss = []
    for seed in seeds:
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'random_hill_climb', max_iters=i, bias=True, is_classifier=True, learning_rate=rhc_lr, restarts=rhc_restarts, early_stopping=False, curve=True, random_state = seed)
        start_time = time.time()
        bp.fit(X_train_scaled, y_train, init_weights = init_weights)
        end_time = time.time()
        temp_time.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_scaled).flatten())
        y_out = pd.Series(bp.predict(X_test_scaled).flatten())
        model_loss = bp.loss
        temp_loss.append(model_loss)
        in_sample_recall = recall_score(y_train, y_in)
        temp_is.append(in_sample_recall)
        out_sample_recall = recall_score(y_test, y_out)
        temp_os.append(out_sample_recall)
    is_recall.append(temp_is)
    os_recall.append(temp_os)
    training_times.append(temp_time)
    training_losses.append(temp_loss)
# Conver(t data into numpy arrays of averages per iterations
is_recall = np.mean(np.array(is_recall), axis=1)
os_recall = np.mean(np.array(os_recall), axis=1)
training_times = np.mean(np.array(training_times), axis=1)
training_losses = np.mean(np.array(training_losses), axis=1)
# Add to overall lists
algo_losses.append(training_losses)
algo_times.append(training_times)
algo_os_recall.append(os_recall)
algo_is_recall.append(is_recall)

# Simulated Annealing
sa_lr = 0.001
sa_sched = mlr.GeomDecay()
is_recall = []
os_recall = []
training_times = []
training_losses = []
for i in iterations:
    temp_is = []
    temp_os = []
    temp_time = []
    temp_loss = []
    for seed in seeds:
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'simulated_annealing', max_iters=i, bias=True, is_classifier=True, learning_rate=sa_lr, schedule=sa_sched, early_stopping=False, curve=True, random_state = seed)
        start_time = time.time()
        bp.fit(X_train_scaled, y_train, init_weights=init_weights)
        end_time = time.time()
        temp_time.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_scaled).flatten())
        y_out = pd.Series(bp.predict(X_test_scaled).flatten())
        model_loss = bp.loss
        temp_loss.append(model_loss)
        in_sample_recall = recall_score(y_train, y_in)
        temp_is.append(in_sample_recall)
        out_sample_recall = recall_score(y_test, y_out)
        temp_os.append(out_sample_recall)
    is_recall.append(temp_is)
    os_recall.append(temp_os)
    training_times.append(temp_time)
    training_losses.append(temp_loss)
# Conver(t data into numpy arrays of averages per iterations
is_recall = np.mean(np.array(is_recall), axis=1)
os_recall = np.mean(np.array(os_recall), axis=1)
training_times = np.mean(np.array(training_times), axis=1)
training_losses = np.mean(np.array(training_losses), axis=1)
# Add to overall lists
algo_losses.append(training_losses)
algo_times.append(training_times)
algo_os_recall.append(os_recall)
algo_is_recall.append(is_recall)

# Genetic Algorithm
ga_lr = 0.0001
ga_prob = 0.1
is_recall = []
os_recall = []
training_times = []
training_losses = []
for i in iterations:
    temp_is = []
    temp_os = []
    temp_time = []
    temp_loss = []
    for seed in seeds:
        init_weights = np.random.uniform(-0.5,0.5,423)
        bp = mlr.NeuralNetwork(hidden_nodes=[13,10,7,5,1], activation='relu', algorithm = 'genetic_alg', max_iters=i, bias=True, is_classifier=True, learning_rate=ga_lr, mutation_prob=ga_prob, pop_size=200, early_stopping=False, curve=True, random_state = seed)
        start_time = time.time()
        bp.fit(X_train_scaled, y_train, init_weights=init_weights)
        end_time = time.time()
        temp_time.append(end_time-start_time)
        y_in = pd.Series(bp.predict(X_train_scaled).flatten())
        y_out = pd.Series(bp.predict(X_test_scaled).flatten())
        model_loss = bp.loss
        temp_loss.append(model_loss)
        in_sample_recall = recall_score(y_train, y_in)
        temp_is.append(in_sample_recall)
        out_sample_recall = recall_score(y_test, y_out)
        temp_os.append(out_sample_recall)
    is_recall.append(temp_is)
    os_recall.append(temp_os)
    training_times.append(temp_time)
    training_losses.append(temp_loss)
# Conver(t data into numpy arrays of averages per iterations
is_recall = np.mean(np.array(is_recall), axis=1)
os_recall = np.mean(np.array(os_recall), axis=1)
training_times = np.mean(np.array(training_times), axis=1)
training_losses = np.mean(np.array(training_losses), axis=1)
# Add to overall lists
algo_losses.append(training_losses)
algo_times.append(training_times)
algo_os_recall.append(os_recall)
algo_is_recall.append(is_recall)

# Create graphs
# Training time
legend = []
for y_vals in algo_times:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(iterations))),iterations)
plt.ylabel('Training Time')
plt.xlabel('Iterations')
plt.title('Comparison of Training Time (Wall Clock)\nPer Iteration for Neural Network')
plt.legend(legend, ['GD', 'RHC', 'SA', 'GA'], title='Algorithm')
plt.savefig('Output/NN_Wall_Clock_Speed.png')
plt.close()

# Training Loss
legend = []
for y_vals in algo_losses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(iterations))),iterations)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.title('Comparison of Training Loss\nPer Iteration for Neural Network')
plt.legend(legend, ['GD', 'RHC', 'SA', 'GA'], title='Algorithm')
plt.savefig('Output/NN_training_loss.png')
plt.close()

# Recall
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Recall Performance by Iteration')
plt.setp(ax1, xticks=range(len(iterations)))
plt.setp(ax1, xticklabels=iterations)
plt.setp(ax2, xticks=range(len(iterations)))
plt.setp(ax2, xticklabels=iterations)
legend_1 = []
for y_vals in algo_is_recall:
    ln, = ax1.plot(y_vals)
    legend_1.append(ln)
ax1.set_title('In Sample Recall Score')
ax1.legend(legend_1, ['GD', 'RHC', 'SA', 'GA'], title='Algorithm', bbox_to_anchor=(1.05,.5))
legend_2 = []
for y_vals in algo_os_recall:
    ln, = ax2.plot(y_vals)
    legend_2.append(ln)
ax2.set_title('Out of Sample Recall Score')
# ax2.legend(legend_2, ['GD', 'RHC', 'SA', 'GA'], title='Algorithm')
plt.tight_layout()
plt.savefig('Output/NN_recall_score.png', bbox_inches='tight')
plt.close()
