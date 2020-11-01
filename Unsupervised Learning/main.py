from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure, recall_score, silhouette_score
from sklearn import mixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import kmeans_v_baseline, kmeans_v_baseline_2, em_v_baseline, em_v_baseline_2, myKMeans, baseline_cluster, myEM, myPCA, myICA, myRCA, myFS, churnNN, churnNN_tuning, NN_plots, NN_comparison_plot
import time
import torch
import pickle

''' CHURN '''

df = pd.read_csv('./Data/churn/Churn_Modeling.csv')

# Preprocessing
df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)

dummies = pd.get_dummies(df[['Geography', 'Gender']])
df.drop(columns=['Geography', 'Gender'],inplace=True)
df = df.merge(dummies, how='inner', left_index=True, right_index=True)

# Split data into train and test
y = df['Exited']
X = df.drop(columns=['Exited'])

# Keep out a larger test set that we won't touch until we get to the NN portion
# We will further split the test sets into a validation and test set for NN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=13)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3, random_state=13)

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

# Scale the X sets
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)


'''
Part 1 Run clustering algorithms on churn data set
'''

'''
KMEANS
 - we will use algorithm = 'full' as this is the traditional EM style algorithm
 
'''

# Dataset 1 - CHURN
kmeans, k, meth, init, iters, kmeans_h, kmeans_c, kmeans_v, kmeans_time = myKMeans(X_train, y_train, 'churn_output_1', 'churn_orig')
pickle.dump(kmeans, open("kmeans.pkl", "wb"))

baseline, base_h, base_c, base_v, base_time = baseline_cluster(X_train, y_train, k, 'churn_output_1', 'churn_orig') # Baseline model made by randomly assigning rows to a cluster

kmeans_v_baseline(kmeans_h, base_h, kmeans_c, base_c, kmeans_v, base_v, kmeans_time, base_time, 'churn_output_1', 'churn_orig')

clusters = kmeans.labels_
kmeans_s_score = silhouette_score(X_train, clusters, metric = 'euclidean')

'''
EM
'''

# Dataset 1 - CHURN
em, em_k, em_meth, em_init, em_iters, em_cov, em_h, em_c, em_v, em_time = myEM(X_train, y_train, 'churn_output', 'churn_orig')
pickle.dump(em, open("em.pkl", "wb"))

em_baseline, em_base_h, em_base_c, em_base_v, em_base_time = baseline_cluster(X_train, y_train, em_k, 'churn_output', 'churn_orig')

em_v_baseline(em_h, em_base_h, em_c, em_base_c, em_v, em_base_v, em_time, em_base_time, 'churn_output', 'churn_orig')

em_s_score = silhouette_score(X_train, em.predict(X_train), metric = 'euclidean')

# Comparison of method speeds
# Dataset 1 - CHURN
colors = ['orange', 'blue']
plt.barh(np.arange(2), [kmeans_time, em_time], color=colors)
plt.yticks(ticks=list(range(2)),labels=['KMeans', 'EM'])
plt.title('KMeans vs EM Running Time')
plt.tight_layout()
plt.savefig('churn_output/kmeans_vs_em.png', bbox_inches='tight')
plt.close()

# Comparison of method performance
colors = ['orange', 'blue']
plt.barh(np.arange(2), [kmeans_s_score, em_s_score], color=colors)
plt.yticks(ticks=list(range(2)),labels=['KMeans', 'EM'])
plt.title('KMeans vs EM Silhouette Score')
plt.tight_layout()
plt.savefig('churn_output/kmeans_vs_em_s_score.png', bbox_inches='tight')
plt.close()

'''
Part 2 Run Dimensionality Reduction algorithms on each data set
  - We can compare the number of components needed through each method
'''

'''
PCA
'''
# Dataset 1 - CHURN
pca, pca_full, pca_time = myPCA(X_train, y_train, 'churn_output_1', 'churn_orig')
pickle.dump(pca, open("pca.pkl", "wb"))

pca_nc = len(pca.explained_variance_) # number of components required according to pca

'''
ICA
'''
# Dataset 1 - CHURN
ica, ica_time = myICA(X_train, y_train, 'churn_output', 'churn_orig')
pickle.dump(ica, open("ica.pkl", "wb"))

ica_nc = ica.n_components

'''
RCA
'''
# Dataset 1 - CHURN
rca, rca_time = myRCA(X_train, y_train, 'churn_output_1', 'churn_orig')
pickle.dump(rca, open("rca.pkl", "wb"))

rca_nc = rca.n_components

'''
RFE
'''
# Dataset 1 - CHURN
fs, fs_time = myFS(X_train, y_train, 'churn_output', 'churn_orig')
pickle.dump(fs, open("fs.pkl", "wb"))

fs_nc = fs.n_features_

# Comparison of method speeds
# Dataset 1 - CHURN
colors = ['orange', 'blue', 'gray', 'black']
plt.barh(np.arange(4), [pca_time, ica_time, rca_time, fs_time], color=colors)
plt.yticks(ticks=list(range(4)),labels=['PCA', 'ICA', 'RCA', 'RFE'])
plt.title('Feature Transformation Method Running Times')
plt.tight_layout()
plt.savefig('churn_output/feature_trans_running_time.png', bbox_inches='tight')
plt.close()

# Comparison of number of components needed for each method
# Dataset 1 - CHURN
colors = ['orange', 'blue', 'gray', 'black']
plt.barh(np.arange(4), [pca_nc, ica_nc, rca_nc, fs_nc], color=colors)
plt.yticks(ticks=list(range(4)),labels=['PCA', 'ICA', 'RCA', 'RFE'])
plt.title('Feature Transformation Method Required Components')
plt.tight_layout()
plt.savefig('churn_output/feature_trans_required_components.png', bbox_inches='tight')
plt.close()

'''
Part 3 Run Clustering algorithms on reduced data sets
'''
# Dataset 1 - CHURN

# transform X_train for each transformation algorithm
pca_X_train = pca.transform(X_train)
ica_X_train = ica.transform(X_train)
rca_X_train = rca.transform(X_train)
fs_X_train = fs.transform(X_train)

# Run KMeans for each of the transformed datasets
kmeans_pca, k_pca, meth_pca, init_pca, iters_pca, kmeans_h_pca, kmeans_c_pca, kmeans_v_pca, kmeans_time_pca = myKMeans(pca_X_train, y_train, 'churn_output', 'churn_pca')
kmeans_ica, k_ica, meth_ica, init_ica, iters_ica, kmeans_h_ica, kmeans_c_ica, kmeans_v_ica, kmeans_time_ica = myKMeans(ica_X_train, y_train, 'churn_output', 'churn_ica')
kmeans_rca, k_rca, meth_rca, init_rca, iters_rca, kmeans_h_rca, kmeans_c_rca, kmeans_v_rca, kmeans_time_rca = myKMeans(rca_X_train, y_train, 'churn_output', 'churn_rca')
kmeans_fs, k_fs, meth_fs, init_fs, iters_fs, kmeans_h_fs, kmeans_c_fs, kmeans_v_fs, kmeans_time_fs = myKMeans(fs_X_train, y_train, 'churn_output', 'churn_fs')

# Run the baseline model for each transformed dataset
baseline_pca, base_h_pca, base_c_pca, base_v_pca, base_time_pca = baseline_cluster(pca_X_train, y_train, k_pca, 'churn_output', 'churn_pca')
baseline_ica, base_h_ica, base_c_ica, base_v_ica, base_time_ica = baseline_cluster(ica_X_train, y_train, k_ica, 'churn_output', 'churn_ica')
baseline_rca, base_h_rca, base_c_rca, base_v_rca, base_time_rca = baseline_cluster(rca_X_train, y_train, k_rca, 'churn_output', 'churn_rca')
baseline_fs, base_h_fs, base_c_fs, base_v_fs, base_time_fs = baseline_cluster(fs_X_train, y_train, k_fs, 'churn_output', 'churn_fs')

# Graph a comparison between each model and the corresponding baseline
kmeans_v_baseline(kmeans_h_pca, base_h_pca, kmeans_c_pca, base_c_pca, kmeans_v_pca, base_v_pca, kmeans_time_pca, base_time_pca, 'churn_output', 'churn_pca')
kmeans_v_baseline(kmeans_h_ica, base_h_ica, kmeans_c_ica, base_c_ica, kmeans_v_ica, base_v_ica, kmeans_time_ica, base_time_ica, 'churn_output', 'churn_ica')
kmeans_v_baseline(kmeans_h_rca, base_h_rca, kmeans_c_rca, base_c_rca, kmeans_v_rca, base_v_rca, kmeans_time_rca, base_time_rca, 'churn_output', 'churn_rca')
kmeans_v_baseline(kmeans_h_fs, base_h_fs, kmeans_c_fs, base_c_fs, kmeans_v_fs, base_v_fs, kmeans_time_fs, base_time_fs, 'churn_output', 'churn_fs')

# Run KMeans on the original dataset using the same k as found for the transformed data set
start_time = time.time()
kmeans_pca_base = KMeans(n_clusters = k_pca, algorithm = 'full', random_state = 13, init = meth_pca, n_init = init_pca).fit(X_train)
kb_pca_t = time.time() - start_time
start_time = time.time()
kmeans_ica_base = KMeans(n_clusters = k_ica, algorithm = 'full', random_state = 13, init = meth_ica, n_init = init_ica).fit(X_train)
kb_ica_t = time.time()-start_time
start_time = time.time()
kmeans_rca_base = KMeans(n_clusters = k_rca, algorithm = 'full', random_state = 13, init = meth_rca, n_init = init_rca).fit(X_train)
kb_rca_t = time.time() - start_time
start_time = time.time()
kmeans_fs_base = KMeans(n_clusters = k_fs, algorithm = 'full', random_state = 13, init = meth_fs, n_init = init_fs).fit(X_train)
kb_fs_t = time.time() - start_time

# Compute metrics for each base kmeans
kb_pca_h, kb_pca_c, kb_pca_v = homogeneity_completeness_v_measure(y_train, kmeans_pca_base.labels_)
kb_ica_h, kb_ica_c, kb_ica_v = homogeneity_completeness_v_measure(y_train, kmeans_ica_base.labels_)
kb_rca_h, kb_rca_c, kb_rca_v = homogeneity_completeness_v_measure(y_train, kmeans_rca_base.labels_)
kb_fs_h, kb_fs_c, kb_fs_v = homogeneity_completeness_v_measure(y_train, kmeans_fs_base.labels_)

# Graph a comparison between each model, transformed kmeans, original kmeans and baseline
kmeans_v_baseline_2(kmeans_h_pca, kb_pca_h, kmeans_h, base_h_pca, kmeans_c_pca, kb_pca_c, kmeans_c, base_c_pca, kmeans_v_pca, kb_pca_v, kmeans_v, base_v_pca, kmeans_time_pca, kb_pca_t, kmeans_time, base_time_pca, 'churn_output', 'churn_pca', 'PCA')
kmeans_v_baseline_2(kmeans_h_ica, kb_ica_h, kmeans_h, base_h_ica, kmeans_c_ica, kb_ica_c, kmeans_c, base_c_ica, kmeans_v_ica, kb_ica_v, kmeans_v, base_v_ica, kmeans_time_ica, kb_ica_t, kmeans_time, base_time_ica, 'churn_output', 'churn_ica', 'ICA')
kmeans_v_baseline_2(kmeans_h_rca, kb_rca_h, kmeans_h, base_h_rca, kmeans_c_rca, kb_rca_c, kmeans_c, base_c_rca, kmeans_v_rca, kb_rca_v, kmeans_v, base_v_rca, kmeans_time_rca, kb_rca_t, kmeans_time, base_time_rca, 'churn_output', 'churn_rca', 'RCA')
kmeans_v_baseline_2(kmeans_h_fs, kb_fs_h, kmeans_h, base_h_fs, kmeans_c_fs, kb_fs_c, kmeans_c, base_c_fs, kmeans_v_fs, kb_fs_v, kmeans_v, base_v_fs, kmeans_time_fs, kb_fs_t, kmeans_time, base_time_fs, 'churn_output', 'churn_fs', 'RFE')

# Run EM for each of the transformed datasets
em_pca, em_k_pca, em_meth_pca, em_init_pca, em_iters_pca, em_cov_pca, em_h_pca, em_c_pca, em_v_pca, em_time_pca = myEM(pca_X_train, y_train, 'churn_output', 'churn_pca')
em_ica, em_k_ica, em_meth_ica, em_init_ica, em_iters_ica, em_cov_ica, em_h_ica, em_c_ica, em_v_ica, em_time_ica = myEM(ica_X_train, y_train, 'churn_output', 'churn_ica')
em_rca, em_k_rca, em_meth_rca, em_init_rca, em_iters_rca, em_cov_rca, em_h_rca, em_c_rca, em_v_rca, em_time_rca = myEM(rca_X_train, y_train, 'churn_output', 'churn_rca')
em_fs, em_k_fs, em_meth_fs, em_init_fs, em_iters_fs, em_cov_fs, em_h_fs, em_c_fs, em_v_fs, em_time_fs = myEM(fs_X_train, y_train, 'churn_output', 'churn_fs')

# Run the baseline model for each transformed dataset
em_baseline_pca, em_base_h_pca, em_base_c_pca, em_base_v_pca, em_base_time_pca = baseline_cluster(pca_X_train, y_train, em_k_pca, 'churn_output', 'churn_pca')
em_baseline_ica, em_base_h_ica, em_base_c_ica, em_base_v_ica, em_base_time_ica = baseline_cluster(ica_X_train, y_train, em_k_ica, 'churn_output', 'churn_ica')
em_baseline_rca, em_base_h_rca, em_base_c_rca, em_base_v_rca, em_base_time_rca = baseline_cluster(rca_X_train, y_train, em_k_rca, 'churn_output', 'churn_rca')
em_baseline_fs, em_base_h_fs, em_base_c_fs, em_base_v_fs, em_base_time_fs = baseline_cluster(fs_X_train, y_train, em_k_fs, 'churn_output', 'churn_fs')

# Graph a comparison between each model and the corresponding baseline
em_v_baseline(em_h_pca, em_base_h_pca, em_c_pca, em_base_c_pca, em_v_pca, em_base_v_pca, em_time_pca, em_base_time_pca, 'churn_output', 'churn_pca')
em_v_baseline(em_h_ica, em_base_h_ica, em_c_ica, em_base_c_ica, em_v_ica, em_base_v_ica, em_time_ica, em_base_time_ica, 'churn_output', 'churn_ica')
em_v_baseline(em_h_rca, em_base_h_rca, em_c_rca, em_base_c_rca, em_v_rca, em_base_v_rca, em_time_rca, em_base_time_rca, 'churn_output', 'churn_rca')
em_v_baseline(em_h_fs, em_base_h_fs, em_c_fs, em_base_c_fs, em_v_fs, em_base_v_fs, em_time_fs, em_base_time_fs, 'churn_output', 'churn_fs')

# Run GMM on the original dataset using the same k as found for the transformed data set
start_time = time.time()
em_pca_base = mixture.GaussianMixture(n_components = em_k_pca, covariance_type = em_cov_pca, max_iter = em_iters_pca, n_init = em_init_pca, init_params = em_meth_pca).fit(X_train)
eb_pca_t = time.time() - start_time
start_time = time.time()
em_ica_base = mixture.GaussianMixture(n_components = em_k_ica, covariance_type = em_cov_ica, max_iter = em_iters_ica, n_init = em_init_ica, init_params = em_meth_ica).fit(X_train)
eb_ica_t = time.time()-start_time
start_time = time.time()
em_rca_base = mixture.GaussianMixture(n_components = em_k_rca, covariance_type = em_cov_rca, max_iter = em_iters_rca, n_init = em_init_rca, init_params = em_meth_rca).fit(X_train)
eb_rca_t = time.time() - start_time
start_time = time.time()
em_fs_base = mixture.GaussianMixture(n_components = em_k_fs, covariance_type = em_cov_fs, max_iter = em_iters_fs, n_init = em_init_fs, init_params = em_meth_fs).fit(X_train)
eb_fs_t = time.time() - start_time

# Compute metrics for each base kmeans
eb_pca_h, eb_pca_c, eb_pca_v = homogeneity_completeness_v_measure(y_train, em_pca_base.predict(X_train))
eb_ica_h, eb_ica_c, eb_ica_v = homogeneity_completeness_v_measure(y_train, em_ica_base.predict(X_train))
eb_rca_h, eb_rca_c, eb_rca_v = homogeneity_completeness_v_measure(y_train, em_rca_base.predict(X_train))
eb_fs_h, eb_fs_c, eb_fs_v = homogeneity_completeness_v_measure(y_train, em_fs_base.predict(X_train))

# Graph a comparison between each model, transformed kmeans, original kmeans and baseline
'''TODO: Fix bar labels'''
em_v_baseline_2(em_h_pca, eb_pca_h, em_h, em_base_h_pca, em_c_pca, eb_pca_c, em_c, em_base_c_pca, em_v_pca, eb_pca_v, em_v, em_base_v_pca, em_time_pca, eb_pca_t, em_time, em_base_time_pca, 'churn_output', 'churn_pca', 'PCA')
em_v_baseline_2(em_h_ica, eb_ica_h, em_h, em_base_h_ica, em_c_ica, eb_ica_c, em_c, em_base_c_ica, em_v_ica, eb_ica_v, em_v, em_base_v_ica, em_time_ica, eb_ica_t, em_time, em_base_time_ica, 'churn_output', 'churn_ica', 'ICA')
em_v_baseline_2(em_h_rca, eb_rca_h, em_h, em_base_h_rca, em_c_rca, eb_rca_c, em_c, em_base_c_rca, em_v_rca, eb_rca_v, em_v, em_base_v_rca, em_time_rca, eb_rca_t, em_time, em_base_time_rca, 'churn_output', 'churn_rca', 'RCA')
em_v_baseline_2(em_h_fs, eb_fs_h, em_h, em_base_h_fs, em_c_fs, eb_fs_c, em_c, em_base_c_fs, em_v_fs, eb_fs_v, em_v, em_base_v_fs, em_time_fs, eb_fs_t, em_time, em_base_time_fs, 'churn_output', 'churn_fs', 'RFE')

'''
Part 4 Run NN on dimensionality reduced data sets
'''

full_recall = []
final_times = []
algos = ['Orig', 'PCA', 'ICA', 'RCA', 'RFE']
alpha = 0.01
gamma = 0.9

# Create various sizes of the testing set
y_train_100 = y_train.head(100)
y_train_1000 = y_train.iloc[100:].head(1000)
y_train_2500 = y_train.iloc[1100:].head(2500)
y_train_5000 = y_train.tail(5000)

# Create various sizes of training set for original data
X_train_100 = X_train.head(100)
X_train_1000 = X_train.iloc[100:].head(1000)
X_train_2500 = X_train.iloc[1100:].head(2500)
X_train_5000 = X_train.tail(5000)

training_sets = [(X_train_100.values, y_train_100.values), (X_train_1000.values, y_train_1000.values), (X_train_2500.values, y_train_2500.values), (X_train_5000.values, y_train_5000.values), (X_train.values, y_train.values)]

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, X_test.values, y_test.values, 'churn_output', 'churn_orig', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)

full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_orig')


# Create various sizes of training set for pca
X_train_100 = pca.transform(X_train.head(100))
X_train_1000 = pca.transform(X_train.iloc[100:].head(1000))
X_train_2500 = pca.transform(X_train.iloc[1100:].head(2500))
X_train_5000 = pca.transform(X_train.tail(5000))

pca_training_sets = [(X_train_100, y_train_100.values), (X_train_1000, y_train_1000.values), (X_train_2500, y_train_2500.values), (X_train_5000, y_train_5000.values), (pca_X_train, y_train.values)]
pca_X_test = pca.transform(X_test)

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(pca_training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in pca_training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, pca_X_test, y_test.values, 'churn_output', 'churn_pca', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)
    
full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_pca')


# Create various sizes of training set for ica
X_train_100 = ica.transform(X_train.head(100))
X_train_1000 = ica.transform(X_train.iloc[100:].head(1000))
X_train_2500 = ica.transform(X_train.iloc[1100:].head(2500))
X_train_5000 = ica.transform(X_train.tail(5000))

ica_training_sets = [(X_train_100, y_train_100.values), (X_train_1000, y_train_1000.values), (X_train_2500, y_train_2500.values), (X_train_5000, y_train_5000.values), (ica_X_train, y_train.values)]
ica_X_test = ica.transform(X_test)

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(ica_training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in ica_training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, ica_X_test, y_test.values, 'churn_output', 'churn_ica', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)
    
full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_ica')
  

# Create various sizes of training set for rca
X_train_100 = rca.transform(X_train.head(100))
X_train_1000 = rca.transform(X_train.iloc[100:].head(1000))
X_train_2500 = rca.transform(X_train.iloc[1100:].head(2500))
X_train_5000 = rca.transform(X_train.tail(5000))

rca_training_sets = [(X_train_100, y_train_100.values), (X_train_1000, y_train_1000.values), (X_train_2500, y_train_2500.values), (X_train_5000, y_train_5000.values), (rca_X_train, y_train.values)]
rca_X_test = rca.transform(X_test)

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(rca_training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in rca_training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, rca_X_test, y_test.values, 'churn_output', 'churn_rca', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)
    
full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_rca')



# Create various sizes of training set for fs
X_train_100 = fs.transform(X_train.head(100))
X_train_1000 = fs.transform(X_train.iloc[100:].head(1000))
X_train_2500 = fs.transform(X_train.iloc[1100:].head(2500))
X_train_5000 = fs.transform(X_train.tail(5000))

fs_training_sets = [(X_train_100, y_train_100.values), (X_train_1000, y_train_1000.values), (X_train_2500, y_train_2500.values), (X_train_5000, y_train_5000.values), (fs_X_train, y_train.values)]
fs_X_test = fs.transform(X_test)

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(pca_training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in fs_training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, fs_X_test, y_test.values, 'churn_output', 'churn_fs', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)
    
full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_fs')

# Create Final Comparison Plot
NN_comparison_plot(full_recall,final_times, algos, 'churn_output', 'churn_all')

'''
Part 5 Run NN on clustered data sets
'''

full_recall = [full_recall[0]]
final_times = [final_times[0]]
algos = ['Orig', 'KMeans', 'EM']

# Create various sizes of the testing set
y_train_100 = y_train.head(100)
y_train_1000 = y_train.iloc[100:].head(1000)
y_train_2500 = y_train.iloc[1100:].head(2500)
y_train_5000 = y_train.tail(5000)

# Create various sizes of training set for kmeans data
X_train_100 = kmeans.transform(X_train.head(100))
X_train_1000 = kmeans.transform(X_train.iloc[100:].head(1000))
X_train_2500 = kmeans.transform(X_train.iloc[1100:].head(2500))
X_train_5000 = kmeans.transform(X_train.tail(5000))
kmeans_X_train = kmeans.transform(X_train)
kmeans_X_test = kmeans.transform(X_test)

kmeans_training_sets = [(X_train_100, y_train_100.values), (X_train_1000, y_train_1000.values), (X_train_2500, y_train_2500.values), (X_train_5000, y_train_5000.values), (kmeans_X_train, y_train.values)]

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in kmeans_training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, kmeans_X_test, y_test.values, 'churn_output', 'churn_kmeans', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)

full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_kmeans')


# Create various sizes of training set for kmeans data
X_train_100 = em.predict_proba(X_train.head(100))
X_train_1000 = em.predict_proba(X_train.iloc[100:].head(1000))
X_train_2500 = em.predict_proba(X_train.iloc[1100:].head(2500))
X_train_5000 = em.predict_proba(X_train.tail(5000))
em_X_train = em.predict_proba(X_train)
em_X_test = em.predict_proba(X_test)

em_training_sets = [(X_train_100, y_train_100.values), (X_train_1000, y_train_1000.values), (X_train_2500, y_train_2500.values), (X_train_5000, y_train_5000.values), (em_X_train, y_train.values)]

#Tune hyper-parameters
#alpha, gamma = churnNN_tuning(training_sets[-1])

# Run NN
training_loss_all = []
in_recalls = []
out_recalls = []
training_times = []
in_time = []
out_time = []
for training_set in em_training_sets:
    training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time = churnNN(training_set, em_X_test, y_test.values, 'churn_output', 'churn_GMM', alpha, gamma)
    training_loss_all.append(training_loss)
    in_recalls.append(in_recall)
    out_recalls.append(out_recall)
    training_times.append(training_time)
    in_time.append(in_query_time)
    out_time.append(out_query_time)

full_recall.append(out_recalls[-1])
final_times.append(training_times[-1])
# Create Plots
plt.close()
NN_plots(training_loss_all, in_recalls, out_recalls, training_times, 'churn_output', 'churn_GMM')
plt.close()
NN_comparison_plot(full_recall, final_times, algos, 'churn_output', 'churn_all_cluster')




''' CHESS '''

df = pd.read_csv('./Data/chess/games.csv')

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
X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

'''
Part 1 Run clustering algorithms on churn data set
'''

'''
KMEANS
 - we will use algorithm = 'full' as this is the traditional EM style algorithm
 
'''

# Dataset 2 - CHESS
kmeans, k, meth, init, iters, kmeans_h, kmeans_c, kmeans_v, kmeans_time = myKMeans(X_train, y_train, 'chess_output', 'chess_orig')
pickle.dump(kmeans, open("kmeans_chess.pkl", "wb"))

baseline, base_h, base_c, base_v, base_time = baseline_cluster(X_train, y_train, k, 'chess_output', 'chess_orig') # Baseline model made by randomly assigning rows to a cluster

kmeans_v_baseline(kmeans_h, base_h, kmeans_c, base_c, kmeans_v, base_v, kmeans_time, base_time, 'chess_output', 'chess_orig')

clusters = kmeans.labels_
kmeans_s_score = silhouette_score(X_train, clusters, metric = 'euclidean')

'''
EM
'''

# Dataset 2 - CHESS
em, em_k, em_meth, em_init, em_iters, em_cov, em_h, em_c, em_v, em_time = myEM(X_train, y_train, 'chess_output', 'chess_orig')
pickle.dump(em, open("em_chess.pkl","wb"))

em_baseline, em_base_h, em_base_c, em_base_v, em_base_time = baseline_cluster(X_train, y_train, em_k, 'chess_output', 'chess_orig')

em_v_baseline(em_h, em_base_h, em_c, em_base_c, em_v, em_base_v, em_time, em_base_time, 'chess_output', 'chess_orig')

em_s_score = silhouette_score(X_train, em.predict(X_train), metric = 'euclidean')

# Comparison of method speeds
# Dataset 2 - CHESS
colors = ['orange', 'blue']
plt.barh(np.arange(2), [kmeans_time, em_time], color=colors)
plt.yticks(ticks=list(range(2)),labels=['KMeans', 'EM'])
plt.title('KMeans vs EM Running Time')
plt.tight_layout()
plt.savefig('chess_output/kmeans_vs_em.png', bbox_inches='tight')
plt.close()

# Comparison of method performance
colors = ['orange', 'blue']
plt.barh(np.arange(2), [kmeans_s_score, em_s_score], color=colors)
plt.yticks(ticks=list(range(2)),labels=['KMeans', 'EM'])
plt.title('KMeans vs EM Silhouette Score')
plt.tight_layout()
plt.savefig('chess_output/kmeans_vs_em_s_score.png', bbox_inches='tight')
plt.close()

'''
Part 2 Run Dimensionality Reduction algorithms on each data set
  - We can compare the number of components needed through each method
'''

'''
PCA
'''
# Dataset 2 - CHESS
pca, pca_full, pca_time = myPCA(X_train, y_train, 'chess_output', 'chess_orig')
pickle.dump(pca, open("pca_chess.pkl","wb"))
pca_nc = len(pca.explained_variance_) # number of components required according to pca

'''
ICA
'''
# Dataset 2 - CHESS
ica, ica_time = myICA(X_train, y_train, 'chess_output', 'chess_orig')
pickle.dump(ica, open("ica_chess.pkl","wb"))
ica_nc = ica.n_components

'''
RCA
'''
# Dataset 2 - CHESS
rca, rca_time = myRCA(X_train, y_train, 'chess_output', 'chess_orig')
pickle.dump(rca, open("rca_chess.pkl","wb"))
rca_nc = rca.n_components

'''
RFE
'''
# Dataset 2 - CHESS
fs, fs_time = myFS(X_train, y_train, 'chess_output', 'chess_orig', 'weighted')
pickle.dump(fs, open("fs_chess.pkl", "wb"))
fs_nc = fs.n_features_

# Comparison of method speeds
# Dataset 2 - CHESS
colors = ['orange', 'blue', 'gray', 'black']
plt.barh(np.arange(4), [pca_time, ica_time, rca_time, fs_time], color=colors)
plt.yticks(ticks=list(range(4)),labels=['PCA', 'ICA', 'RCA', 'RFE'])
plt.title('Feature Transformation Method Running Times')
plt.tight_layout()
plt.savefig('chess_output/feature_trans_running_time.png', bbox_inches='tight')
plt.close()

# Comparison of number of components needed for each method
# Dataset 2 - CHESS
colors = ['orange', 'blue', 'gray', 'black']
plt.barh(np.arange(4), [pca_nc, ica_nc, rca_nc, fs_nc], color=colors)
plt.yticks(ticks=list(range(4)),labels=['PCA', 'ICA', 'RCA', 'RFE'])
plt.title('Feature Transformation Method Required Components')
plt.tight_layout()
plt.savefig('chess_output/feature_trans_required_components.png', bbox_inches='tight')
plt.close()

'''
Part 3 Run Clustering algorithms on reduced data sets
'''
# Dataset 2 - CHESS

# transform X_train for each transformation algorithm
pca_X_train = pca.transform(X_train)
ica_X_train = ica.transform(X_train)
rca_X_train = rca.transform(X_train)
fs_X_train = fs.transform(X_train)

# Run KMeans for each of the transformed datasets
kmeans_pca, k_pca, meth_pca, init_pca, iters_pca, kmeans_h_pca, kmeans_c_pca, kmeans_v_pca, kmeans_time_pca = myKMeans(pca_X_train, y_train, 'chess_output', 'chess_pca')
kmeans_ica, k_ica, meth_ica, init_ica, iters_ica, kmeans_h_ica, kmeans_c_ica, kmeans_v_ica, kmeans_time_ica = myKMeans(ica_X_train, y_train, 'chess_output', 'chess_ica')
kmeans_rca, k_rca, meth_rca, init_rca, iters_rca, kmeans_h_rca, kmeans_c_rca, kmeans_v_rca, kmeans_time_rca = myKMeans(rca_X_train, y_train, 'chess_output', 'chess_rca')
kmeans_fs, k_fs, meth_fs, init_fs, iters_fs, kmeans_h_fs, kmeans_c_fs, kmeans_v_fs, kmeans_time_fs = myKMeans(fs_X_train, y_train, 'chess_output', 'chess_fs')

# Run the baseline model for each transformed dataset
baseline_pca, base_h_pca, base_c_pca, base_v_pca, base_time_pca = baseline_cluster(pca_X_train, y_train, k_pca, 'chess_output', 'chess_pca')
baseline_ica, base_h_ica, base_c_ica, base_v_ica, base_time_ica = baseline_cluster(ica_X_train, y_train, k_ica, 'chess_output', 'chess_ica')
baseline_rca, base_h_rca, base_c_rca, base_v_rca, base_time_rca = baseline_cluster(rca_X_train, y_train, k_rca, 'chess_output', 'chess_rca')
baseline_fs, base_h_fs, base_c_fs, base_v_fs, base_time_fs = baseline_cluster(fs_X_train, y_train, k_fs, 'chess_output', 'chess_fs')

# Graph a comparison between each model and the corresponding baseline
kmeans_v_baseline(kmeans_h_pca, base_h_pca, kmeans_c_pca, base_c_pca, kmeans_v_pca, base_v_pca, kmeans_time_pca, base_time_pca, 'chess_output', 'chess_pca')
kmeans_v_baseline(kmeans_h_ica, base_h_ica, kmeans_c_ica, base_c_ica, kmeans_v_ica, base_v_ica, kmeans_time_ica, base_time_ica, 'chess_output', 'chess_ica')
kmeans_v_baseline(kmeans_h_rca, base_h_rca, kmeans_c_rca, base_c_rca, kmeans_v_rca, base_v_rca, kmeans_time_rca, base_time_rca, 'chess_output', 'chess_rca')
kmeans_v_baseline(kmeans_h_fs, base_h_fs, kmeans_c_fs, base_c_fs, kmeans_v_fs, base_v_fs, kmeans_time_fs, base_time_fs, 'chess_output', 'chess_fs')

# Run KMeans on the original dataset using the same k as found for the transformed data set
start_time = time.time()
kmeans_pca_base = KMeans(n_clusters = k_pca, algorithm = 'full', random_state = 13, init = meth_pca, n_init = init_pca).fit(X_train)
kb_pca_t = time.time() - start_time
start_time = time.time()
kmeans_ica_base = KMeans(n_clusters = k_ica, algorithm = 'full', random_state = 13, init = meth_ica, n_init = init_ica).fit(X_train)
kb_ica_t = time.time()-start_time
start_time = time.time()
kmeans_rca_base = KMeans(n_clusters = k_rca, algorithm = 'full', random_state = 13, init = meth_rca, n_init = init_rca).fit(X_train)
kb_rca_t = time.time() - start_time
start_time = time.time()
kmeans_fs_base = KMeans(n_clusters = k_fs, algorithm = 'full', random_state = 13, init = meth_fs, n_init = init_fs).fit(X_train)
kb_fs_t = time.time() - start_time

# Compute metrics for each base kmeans
kb_pca_h, kb_pca_c, kb_pca_v = homogeneity_completeness_v_measure(y_train, kmeans_pca_base.labels_)
kb_ica_h, kb_ica_c, kb_ica_v = homogeneity_completeness_v_measure(y_train, kmeans_ica_base.labels_)
kb_rca_h, kb_rca_c, kb_rca_v = homogeneity_completeness_v_measure(y_train, kmeans_rca_base.labels_)
kb_fs_h, kb_fs_c, kb_fs_v = homogeneity_completeness_v_measure(y_train, kmeans_fs_base.labels_)

# Graph a comparison between each model, transformed kmeans, original kmeans and baseline
kmeans_v_baseline_2(kmeans_h_pca, kb_pca_h, kmeans_h, base_h_pca, kmeans_c_pca, kb_pca_c, kmeans_c, base_c_pca, kmeans_v_pca, kb_pca_v, kmeans_v, base_v_pca, kmeans_time_pca, kb_pca_t, kmeans_time, base_time_pca, 'chess_output', 'chess_pca', 'PCA')
kmeans_v_baseline_2(kmeans_h_ica, kb_ica_h, kmeans_h, base_h_ica, kmeans_c_ica, kb_ica_c, kmeans_c, base_c_ica, kmeans_v_ica, kb_ica_v, kmeans_v, base_v_ica, kmeans_time_ica, kb_ica_t, kmeans_time, base_time_ica, 'chess_output', 'chess_ica', 'ICA')
kmeans_v_baseline_2(kmeans_h_rca, kb_rca_h, kmeans_h, base_h_rca, kmeans_c_rca, kb_rca_c, kmeans_c, base_c_rca, kmeans_v_rca, kb_rca_v, kmeans_v, base_v_rca, kmeans_time_rca, kb_rca_t, kmeans_time, base_time_rca, 'chess_output', 'chess_rca', 'RCA')
kmeans_v_baseline_2(kmeans_h_fs, kb_fs_h, kmeans_h, base_h_fs, kmeans_c_fs, kb_fs_c, kmeans_c, base_c_fs, kmeans_v_fs, kb_fs_v, kmeans_v, base_v_fs, kmeans_time_fs, kb_fs_t, kmeans_time, base_time_fs, 'chess_output', 'chess_fs', 'RFE')

# Run EM for each of the transformed datasets
em_pca, em_k_pca, em_meth_pca, em_init_pca, em_iters_pca, em_cov_pca, em_h_pca, em_c_pca, em_v_pca, em_time_pca = myEM(pca_X_train, y_train, 'chess_output', 'chess_pca')
em_ica, em_k_ica, em_meth_ica, em_init_ica, em_iters_ica, em_cov_ica, em_h_ica, em_c_ica, em_v_ica, em_time_ica = myEM(ica_X_train, y_train, 'chess_output', 'chess_ica')
em_rca, em_k_rca, em_meth_rca, em_init_rca, em_iters_rca, em_cov_rca, em_h_rca, em_c_rca, em_v_rca, em_time_rca = myEM(rca_X_train, y_train, 'chess_output', 'chess_rca')
em_fs, em_k_fs, em_meth_fs, em_init_fs, em_iters_fs, em_cov_fs, em_h_fs, em_c_fs, em_v_fs, em_time_fs = myEM(fs_X_train, y_train, 'chess_output', 'chess_fs')

# Run the baseline model for each transformed dataset
em_baseline_pca, em_base_h_pca, em_base_c_pca, em_base_v_pca, em_base_time_pca = baseline_cluster(pca_X_train, y_train, em_k_pca, 'chess_output', 'chess_pca')
em_baseline_ica, em_base_h_ica, em_base_c_ica, em_base_v_ica, em_base_time_ica = baseline_cluster(ica_X_train, y_train, em_k_ica, 'chess_output', 'chess_ica')
em_baseline_rca, em_base_h_rca, em_base_c_rca, em_base_v_rca, em_base_time_rca = baseline_cluster(rca_X_train, y_train, em_k_rca, 'chess_output', 'chess_rca')
em_baseline_fs, em_base_h_fs, em_base_c_fs, em_base_v_fs, em_base_time_fs = baseline_cluster(fs_X_train, y_train, em_k_fs, 'chess_output', 'chess_fs')

# Graph a comparison between each model and the corresponding baseline
em_v_baseline(em_h_pca, em_base_h_pca, em_c_pca, em_base_c_pca, em_v_pca, em_base_v_pca, em_time_pca, em_base_time_pca, 'chess_output', 'chess_pca')
em_v_baseline(em_h_ica, em_base_h_ica, em_c_ica, em_base_c_ica, em_v_ica, em_base_v_ica, em_time_ica, em_base_time_ica, 'chess_output', 'chess_ica')
em_v_baseline(em_h_rca, em_base_h_rca, em_c_rca, em_base_c_rca, em_v_rca, em_base_v_rca, em_time_rca, em_base_time_rca, 'chess_output', 'chess_rca')
em_v_baseline(em_h_fs, em_base_h_fs, em_c_fs, em_base_c_fs, em_v_fs, em_base_v_fs, em_time_fs, em_base_time_fs, 'chess_output', 'chess_fs')

# Run GMM on the original dataset using the same k as found for the transformed data set
start_time = time.time()
em_pca_base = mixture.GaussianMixture(n_components = em_k_pca, covariance_type = em_cov_pca, max_iter = em_iters_pca, n_init = em_init_pca, init_params = em_meth_pca).fit(X_train)
eb_pca_t = time.time() - start_time
start_time = time.time()
em_ica_base = mixture.GaussianMixture(n_components = em_k_ica, covariance_type = em_cov_ica, max_iter = em_iters_ica, n_init = em_init_ica, init_params = em_meth_ica).fit(X_train)
eb_ica_t = time.time()-start_time
start_time = time.time()
em_rca_base = mixture.GaussianMixture(n_components = em_k_rca, covariance_type = em_cov_rca, max_iter = em_iters_rca, n_init = em_init_rca, init_params = em_meth_rca).fit(X_train)
eb_rca_t = time.time() - start_time
start_time = time.time()
em_fs_base = mixture.GaussianMixture(n_components = em_k_fs, covariance_type = em_cov_fs, max_iter = em_iters_fs, n_init = em_init_fs, init_params = em_meth_fs).fit(X_train)
eb_fs_t = time.time() - start_time

# Compute metrics for each base kmeans
eb_pca_h, eb_pca_c, eb_pca_v = homogeneity_completeness_v_measure(y_train, em_pca_base.predict(X_train))
eb_ica_h, eb_ica_c, eb_ica_v = homogeneity_completeness_v_measure(y_train, em_ica_base.predict(X_train))
eb_rca_h, eb_rca_c, eb_rca_v = homogeneity_completeness_v_measure(y_train, em_rca_base.predict(X_train))
eb_fs_h, eb_fs_c, eb_fs_v = homogeneity_completeness_v_measure(y_train, em_fs_base.predict(X_train))

# Graph a comparison between each model, transformed kmeans, original kmeans and baseline
'''TODO: Fix bar labels'''
em_v_baseline_2(em_h_pca, eb_pca_h, em_h, em_base_h_pca, em_c_pca, eb_pca_c, em_c, em_base_c_pca, em_v_pca, eb_pca_v, em_v, em_base_v_pca, em_time_pca, eb_pca_t, em_time, em_base_time_pca, 'chess_output', 'chess_pca', 'PCA')
em_v_baseline_2(em_h_ica, eb_ica_h, em_h, em_base_h_ica, em_c_ica, eb_ica_c, em_c, em_base_c_ica, em_v_ica, eb_ica_v, em_v, em_base_v_ica, em_time_ica, eb_ica_t, em_time, em_base_time_ica, 'chess_output', 'chess_ica', 'ICA')
em_v_baseline_2(em_h_rca, eb_rca_h, em_h, em_base_h_rca, em_c_rca, eb_rca_c, em_c, em_base_c_rca, em_v_rca, eb_rca_v, em_v, em_base_v_rca, em_time_rca, eb_rca_t, em_time, em_base_time_rca, 'chess_output', 'chess_rca', 'RCA')
em_v_baseline_2(em_h_fs, eb_fs_h, em_h, em_base_h_fs, em_c_fs, eb_fs_c, em_c, em_base_c_fs, em_v_fs, eb_fs_v, em_v, em_base_v_fs, em_time_fs, eb_fs_t, em_time, em_base_time_fs, 'chess_output', 'chess_fs', 'RFE')
