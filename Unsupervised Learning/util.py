from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import kurtosis, mode
import time
import torch

def MSE(A,B):
    return (np.square(A-B)).mean(axis=None)

def kmeans_v_baseline(kmeans_h, base_h, kmeans_c, base_c, kmeans_v, base_v, kmeans_t, base_t, output_folder, experiment_name):
    colors = ['blue', 'grey']
    fig, axs = plt.subplots(2,2)
    h = [kmeans_h, base_h]
    axs[0,0].barh(np.arange(len(h)), h, color=colors)
    axs[0,0].set_yticks(ticks=list(range(len(h))))
    axs[0,0].set_yticklabels(['KMeans', 'Baseline'])
    axs[0,0].set_title('Homogeneity Score')
    c = [kmeans_c, base_c]
    axs[0,1].barh(np.arange(len(c)), c, color=colors)
    axs[0,1].set_yticks(ticks=list(range(len(c))))
    axs[0,1].set_yticklabels(['KMeans', 'Baseline'])
    axs[0,1].set_title('Completeness Score')
    v = [kmeans_v, base_v]
    axs[1,0].barh(np.arange(len(v)), v, color=colors)
    axs[1,0].set_yticks(ticks=list(range(len(v))))
    axs[1,0].set_yticklabels(['KMeans', 'Baseline'])
    axs[1,0].set_title('V Score')
    t = [kmeans_t, base_t]
    axs[1,1].barh(np.arange(len(t)), t, color=colors)
    axs[1,1].set_yticks(ticks=list(range(len(t))))
    axs[1,1].set_yticklabels(['KMeans', 'Baseline'])
    axs[1,1].set_title('Running Time')
    fig.suptitle('KMeans Performance vs Random Baseline')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_folder+'/'+experiment_name+'_kmeans_v_baseline.png', bbox_inches='tight')
    plt.close()
    
def kmeans_v_baseline_2(fse_h, kmeans_base_h, kmeans_h, base_h, fse_c, kmeans_base_c, kmeans_c, base_c, fse_v, kmeans_base_v, kmeans_v, base_v, fse_t, kmeans_base_t, kmeans_t, base_t, output_folder, experiment_name, trans_name):
    colors = ['blue', 'grey', 'grey', 'grey']
    fig, axs = plt.subplots(2,2)
    h = [fse_h, kmeans_base_h, kmeans_h, base_h]
    axs[0,0].barh(np.arange(len(h)), h, color=colors)
    axs[0,0].set_yticks(ticks=list(range(len(h))))
    axs[0,0].set_yticklabels([trans_name, 'KMean-Base', 'KMeans', 'Baseline'])
    axs[0,0].set_title('Homogeneity Score')
    c = [fse_c, kmeans_base_c, kmeans_c, base_c]
    axs[0,1].barh(np.arange(len(c)), c, color=colors)
    axs[0,1].set_yticks(ticks=list(range(len(c))))
    axs[0,1].set_yticklabels([trans_name, 'KMean-Base', 'KMeans', 'Baseline'])
    axs[0,1].set_title('Completeness Score')
    v = [fse_v, kmeans_base_v, kmeans_v, base_v]
    axs[1,0].barh(np.arange(len(v)), v, color=colors)
    axs[1,0].set_yticks(ticks=list(range(len(v))))
    axs[1,0].set_yticklabels([trans_name, 'KMean-Base', 'KMeans', 'Baseline'])
    axs[1,0].set_title('V Score')
    t = [fse_t, kmeans_base_t, kmeans_t, base_t]
    axs[1,1].barh(np.arange(len(t)), t, color=colors)
    axs[1,1].set_yticks(ticks=list(range(len(t))))
    axs[1,1].set_yticklabels([trans_name, 'KMean-Base', 'KMeans', 'Baseline'])
    axs[1,1].set_title('Running Time')
    fig.suptitle('Performance Comparison of Different Algorithms')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_folder+'/'+experiment_name+'_kmeans_v_baseline_2.png', bbox_inches='tight')
    plt.close()
    
def em_v_baseline(em_h, em_base_h, em_c, em_base_c, em_v, em_base_v, em_t, base_t, output_folder, experiment_name):
    colors = ['blue', 'grey'] 
    fig, axs = plt.subplots(2,2)
    h = [em_h, em_base_h]
    axs[0,0].barh(np.arange(len(h)), h, color=colors)
    axs[0,0].set_yticks(ticks=list(range(len(h))))
    axs[0,0].set_yticklabels(['EM', 'Baseline'])
    axs[0,0].set_title('Homogeneity Score')
    c = [em_c, em_base_c]
    axs[0,1].barh(np.arange(len(c)), c, color=colors)
    axs[0,1].set_yticks(ticks=list(range(len(c))))
    axs[0,1].set_yticklabels(['EM', 'Baseline'])
    axs[0,1].set_title('Completeness Score')
    v = [em_v, em_base_v]
    axs[1,0].barh(np.arange(len(v)), v, color=colors)
    axs[1,0].set_yticks(ticks=list(range(len(v))))
    axs[1,0].set_yticklabels(['EM', 'Baseline'])
    axs[1,0].set_title('V Score')
    t = [em_t, base_t]
    axs[1,1].barh(np.arange(len(t)), t, color=colors)
    axs[1,1].set_yticks(ticks=list(range(len(t))))
    axs[1,1].set_yticklabels(['EM', 'Baseline'])
    axs[1,1].set_title('Running Time')
    fig.suptitle('EM Performance vs Random Baseline')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_folder+'/'+experiment_name+'_GMM_vs_baseline.png', bbox_inches='tight')
    plt.close()
    
def em_v_baseline_2(fse_h, em_base_h, em_h, base_h, fse_c, em_base_c, em_c, base_c, fse_v, em_base_v, em_v, base_v, fse_t, em_base_t, em_t, base_t, output_folder, experiment_name, trans_name):
    colors = ['blue', 'grey', 'grey', 'grey']
    fig, axs = plt.subplots(2,2)
    h = [fse_h, em_base_h, em_h, base_h]
    axs[0,0].barh(np.arange(len(h)), h, color=colors)
    axs[0,0].set_yticks(ticks=list(range(len(h))))
    axs[0,0].set_yticklabels([trans_name, 'EM-Base', 'EM', 'Baseline'])
    axs[0,0].set_title('Homogeneity Score')
    c = [fse_c, em_base_c, em_c, base_c]
    axs[0,1].barh(np.arange(len(c)), c, color=colors)
    axs[0,1].set_yticks(ticks=list(range(len(c))))
    axs[0,1].set_yticklabels([trans_name, 'EM-Base', 'KMeans', 'Baseline'])
    axs[0,1].set_title('Completeness Score')
    v = [fse_v, em_base_v, em_v, base_v]
    axs[1,0].barh(np.arange(len(v)), v, color=colors)
    axs[1,0].set_yticks(ticks=list(range(len(v))))
    axs[1,0].set_yticklabels([trans_name, 'EM-Base', 'KMeans', 'Baseline'])
    axs[1,0].set_title('V Score')
    t = [fse_t, em_base_t, em_t, base_t]
    axs[1,1].barh(np.arange(len(t)), t, color=colors)
    axs[1,1].set_yticks(ticks=list(range(len(t))))
    axs[1,1].set_yticklabels([trans_name, 'EM-Base', 'KMeans', 'Baseline'])
    axs[1,1].set_title('Running Time')
    fig.suptitle('Performance Comparison of Different Algorithms')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_folder+'/'+experiment_name+'_GMM_v_baseline_2.png', bbox_inches='tight')
    plt.close()

def baseline_cluster(data, act_labels, k, output_folder, experiment_name):
    start_time = time.time()
    clusters = np.random.randint(0,k, size=data.shape[0])
    end_time = time.time()
    final_time = end_time - start_time
    h, c, v = homogeneity_completeness_v_measure(act_labels, clusters)
    return clusters, h, c, v, final_time

def myKMeans(data, act_labels, output_folder, experiment_name):
    # Create random seed list
    rn = np.random.RandomState(13)
    random_seeds = list(rn.randint(1,1000000,1))
    
    # What value of k makes the most sense - use the silhouette score (choose the value that maximizes the silhouette score)   
    s_score = []
    sse = []
    k_values = [2,3,4,5,6,7,8,9]
    for k in k_values:
        s_score_temp = []
        sse_temp = []
        for r in random_seeds:
            kmeans = KMeans(n_clusters = k, algorithm='full', random_state = r).fit(data) 
            labels = kmeans.labels_
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
            sse_temp.append(kmeans.inertia_)
        s_score.append(np.mean(s_score_temp))
        sse.append(np.mean(sse_temp))
        
    keep_k = k_values[np.argmax(s_score)]
    
    plt.plot(s_score)
    plt.xticks(ticks=list(range(len(k_values))), labels=k_values)
    plt.xlabel('K Value')
    plt.ylabel('Silhouette Score')
    plt.title('Average Silhouette Score for Various K Values')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_k_s_score.png')
    plt.close()
    plt.figure()
    
    # Lets also plot the SSE for each value of K
    plt.plot(sse)
    plt.xticks(ticks=list(range(len(k_values))), labels=k_values)
    plt.xlabel('K Value')
    plt.ylabel('SSE')
    plt.title('Average SSE for Various K Values')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_k_sse.png')
    plt.close()
    plt.figure()
    
    # Let's see how the silhouette score changes if we change the way we initialize centers
    s_score = []
    init_meths = ['k-means++', 'random']
    for init in init_meths:
        s_score_temp = []
        for r in random_seeds:
            kmeans = KMeans(n_clusters = keep_k, algorithm='full', random_state=r, init=init).fit(data)
            labels = kmeans.labels_
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        s_score.append(np.mean(s_score_temp))
    
    keep_meth = init_meths[np.argmax(s_score)]
    
    plt.barh(np.arange(len(s_score)), s_score)
    plt.yticks(ticks=list(range(len(init_meths))), labels=init_meths)
    plt.xlabel('Silhouette Score')
    plt.ylabel('Init Method')
    plt.title('Average Silhouette Score for Initialization Methods')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_init_s_score.png')
    plt.close()
    plt.figure()
    
    # Now let's see how silhouette score changes based on the number of times we initialize new centers and run
    s_score = []
    inits = [10, 50, 100, 250, 500, 1000]
    for i in inits:
        kmeans = KMeans(n_clusters = keep_k, init=keep_meth, algorithm = 'full', random_state=13, n_init = i).fit(data)
        labels = kmeans.labels_
        s_score.append(silhouette_score(data, labels, metric = 'euclidean'))
        
    keep_init = inits[np.argmax(s_score)]
    
    plt.plot(s_score)
    plt.xticks(ticks=list(range(len(inits))), labels=inits)
    plt.xlabel('Initializations')
    plt.ylabel('Silhouette Score')
    plt.title('Average Silhouette Score for Various Initializations')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_init2_s_score.png')
    plt.close()
    plt.figure()
    
    # Finally lets tune the number of iterations to perform
    s_score = []
    iters = [300, 500, 800, 1000]
    for i in iters:
        s_score_temp = []
        for r in random_seeds:
            kmeans = KMeans(n_clusters = keep_k, init=keep_meth, n_init=keep_init, algorithm = 'full', random_state=r, max_iter = i).fit(data)
            labels = kmeans.labels_
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        s_score.append(np.mean(s_score_temp))
    
    keep_iter = iters[np.argmax(s_score)]
    
    plt.plot(s_score)
    plt.xticks(ticks=list(range(len(iters))), labels=iters)
    plt.xlabel('Iterations')
    plt.ylabel('Silhouette Score')
    plt.title('Average Silhouette Score for Various Iterations')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_iter_s_score.png')
    plt.close()
    plt.figure()
    
    # Run KMeans with the chosen values for hyper-parameters
    start_time = time.time()
    kmeans = KMeans(n_clusters = keep_k, algorithm = 'full', random_state = 13, init = keep_meth, n_init = keep_init, max_iter=keep_iter).fit(data)
    end_time = time.time()
    final_time = end_time - start_time
    
    # Plot Cluster Cardinality
    cardinality = np.bincount(kmeans.labels_)
    clusters = np.unique(kmeans.labels_)
    plt.bar(clusters, cardinality)
    plt.xlabel('Cluster')
    plt.ylabel('Cardinality')
    plt.title('Cluster Cardinality')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_cardinality.png')
    plt.close()
    
    # Plot Cluster Magnitude
    cluster_labels = kmeans.labels_
    distances = pd.DataFrame(kmeans.transform(data))
    center_distances = pd.DataFrame(np.zeros(distances.shape))
    for index, row in distances.iterrows():
        lab = cluster_labels[index]
        val = np.min(row.values)
        center_distances.iloc[index,lab]=val
    
    center_distances = center_distances.values
    magnitudes = np.sum(center_distances, axis=0)
    plt.bar(clusters, magnitudes)
    plt.xlabel('Cluster')
    plt.ylabel('Magnitude')
    plt.title('Cluster Magnitude')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_magnitude.png')
    plt.close()
    
    # Plot magnitude vs cardinality
    plt.scatter(cardinality, magnitudes)
    plt.xlabel('Cardinality')
    plt.ylabel('Magnitude')
    plt.title('Cardinality x Magnitude')
    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_card_x_mag.png')    
    plt.close
    
    # Evaluate the clusters by using homogeneity, completeness and v score
    h, c, v = homogeneity_completeness_v_measure(act_labels, kmeans.labels_)
    return kmeans, keep_k, keep_meth, keep_init, keep_iter, h, c, v, final_time

def myEM(data, act_labels, output_folder, experiment_name):
    # Create random seed list
    rn = np.random.RandomState(13)
    random_seeds = list(rn.randint(1,1000000,20))
    
    # What value of k makes the most sense - use the BIC
    bics = []
    s_score = []
    k_values = [2,3,4,5,6,7,8,9]
    for k in k_values:
        bics_temp = []
        s_score_temp = []
        for r in random_seeds:
            em = mixture.GaussianMixture(n_components = k, random_state = r).fit(data) 
            bics_temp.append(em.bic(data))
            labels = em.predict(data)
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        bics.append(np.mean(bics_temp))
        s_score.append(np.mean(s_score_temp))
        
    keep_k = k_values[np.argmin(bics)]
    
    plt.plot(bics)
    plt.xticks(ticks=list(range(len(k_values))), labels=k_values)
    plt.xlabel('K Value')
    plt.ylabel('BIC')
    plt.title('Average BIC for Various K Values')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_k_BIC.png')
    plt.close()
    plt.figure()
    
    plt.plot(s_score)
    plt.xticks(ticks=list(range(len(k_values))), labels=k_values)
    plt.xlabel('K Value')
    plt.ylabel('Silhouette Score')
    plt.title('Average BIC for Various K Values')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_k_s_score.png')
    plt.close()
    plt.figure()
    
    
    # Let's see how the BIC changes if we change the way we initialize centers
    bics = []
    s_score = []
    init_meths = ['kmeans', 'random']
    for init in init_meths:
        bics_temp = []
        s_score_temp = []
        for r in random_seeds:
            em = mixture.GaussianMixture(n_components = keep_k, random_state=r, init_params=init).fit(data)
            bics_temp.append(em.bic(data))
            labels = em.predict(data)
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        bics.append(np.mean(bics_temp))
        s_score.append(np.mean(s_score_temp))
    
    keep_meth = init_meths[np.argmin(bics)]
    
    plt.barh(np.arange(len(bics)), bics)
    plt.yticks(ticks=list(range(len(init_meths))), labels=init_meths)
    plt.xlabel('BIC')
    plt.ylabel('Init Method')
    plt.title('Average BIC for Initialization Methods')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_init_BIC.png')
    plt.close()
    plt.figure()
    
    plt.barh(np.arange(len(s_score)), s_score)
    plt.yticks(ticks=list(range(len(init_meths))), labels=init_meths)
    plt.xlabel('Silhouette Score')
    plt.ylabel('Init Method')
    plt.title('Average Silhouette Score for Initialization Methods')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_init_s_score.png')
    plt.close()
    plt.figure()
    
    
    # Now let's see how BIC changes based on the number of times we initialize new centers and run
    bics = []
    s_score = []
    inits = [10, 50, 100, 250, 500, 1000]
    for i in inits:
        bics_temp = []
        s_score_temp = []
        for r in random_seeds:
            em = mixture.GaussianMixture(n_components = keep_k, random_state=r, n_init = i).fit(data)
            bics_temp.append(em.bic(data))
            labels = em.predict(data)
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        bics.append(np.mean(bics_temp))
        s_score.append(np.mean(s_score_temp))
        
    keep_init = inits[np.argmin(bics)]
    
    plt.plot(bics)
    plt.xticks(ticks=list(range(len(inits))), labels=inits)
    plt.xlabel('Initializations')
    plt.ylabel('BIC')
    plt.title('Average BIC for Various Initializations')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_init2_BIC.png')
    plt.close()
    plt.figure()    
    
    plt.plot(s_score)
    plt.xticks(ticks=list(range(len(inits))), labels=inits)
    plt.xlabel('Initializations')
    plt.ylabel('Silhouette Score')
    plt.title('Average Silhouette Score for Various Initializations')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_init2_s_score.png')
    plt.close()
    plt.figure()
    
    
    # Tune Covariance type using BIC
    bics = []
    s_score = []
    covs = ['full', 'tied', 'diag', 'spherical']
    for c in covs:
        bics_temp = []
        s_score_temp = []
        for r in random_seeds:
            em = mixture.GaussianMixture(n_components = keep_k, random_state=r, covariance_type=c).fit(data)
            bics_temp.append(em.bic(data))
            labels = em.predict(data)
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        bics.append(np.mean(bics_temp))
        s_score.append(np.mean(s_score_temp))
    
    keep_cov = covs[np.argmin(bics)]
    
    plt.barh(np.arange(len(s_score)), s_score)
    plt.yticks(ticks=list(range(len(covs))), labels=covs)
    plt.xlabel('BIC')
    plt.ylabel('Covariance Type')
    plt.title('Average BIC for Covariance Types')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_cov_BIC.png')
    plt.close()
    plt.figure()
    
    plt.barh(np.arange(len(s_score)), s_score)
    plt.yticks(ticks=list(range(len(covs))), labels=covs)
    plt.xlabel('Silhouette Score')
    plt.ylabel('Covariance Type')
    plt.title('Average Silhouette Score for Covariance Types')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_cov_s_score.png')
    plt.close()
    plt.figure()
    
    
    # Finally lets tune the number of iterations to perform based on BICS
    bics = []
    s_score = []
    iters = [300, 500, 800, 1000]
    for i in iters:
        bics_temp = []
        s_score_temp = []
        for r in random_seeds:
            em = mixture.GaussianMixture(n_components = keep_k, random_state=r, max_iter = i).fit(data)
            bics_temp.append(em.bic(data))
            labels = em.predict(data)
            s_score_temp.append(silhouette_score(data, labels, metric = 'euclidean'))
        bics.append(np.mean(bics_temp))
        s_score.append(np.mean(s_score_temp))
    
    keep_iter = iters[np.argmax(s_score)]
    
    plt.plot(bics)
    plt.xticks(ticks=list(range(len(iters))), labels=iters)
    plt.xlabel('Iterations')
    plt.ylabel('BIC')
    plt.title('Average Silhouette Score for Various Iterations')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_iter_BIC.png')
    plt.close()
    plt.figure()

    plt.plot(s_score)
    plt.xticks(ticks=list(range(len(iters))), labels=iters)
    plt.xlabel('Iterations')
    plt.ylabel('Silhouette Score')
    plt.title('Average Silhouette Score for Various Iterations')
    plt.savefig(output_folder+'/'+experiment_name+'_GMM_iter_s_score.png')
    plt.close()
    plt.figure()
    
    # Run GMM with the chosen values for hyper-parameters
    start_time = time.time()
    em = mixture.GaussianMixture(n_components = keep_k, covariance_type = keep_cov, max_iter = keep_iter, n_init = keep_init, init_params = keep_meth).fit(data)
    end_time = time.time()
    final_time = end_time - start_time
    
#    # Plot Cluster Cardinality
#    cardinality = np.bincount(em.predict(data))
#    clusters = np.unique(em.predict(data))
#    plt.bar(clusters, cardinality)
#    plt.xlabel('Cluster')
#    plt.ylabel('Cardinality')
#    plt.title('Cluster Cardinality')
#    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_cardinality.png')
#    
#    # Plot Cluster Magnitude
#    cluster_labels = em.predict(data)
#    distances = pd.DataFrame(em.transform(data))
#    center_distances = pd.DataFrame(np.zeros(distances.shape))
#    for index, row in distances.iterrows():
#        lab = cluster_labels[index]
#        val = np.min(row.values)
#        center_distances.iloc[index,lab]=val
#    
#    center_distances = center_distances.values
#    magnitudes = np.sum(center_distances, axis=0)
#    plt.bar(clusters, magnitudes)
#    plt.xlabel('Cluster')
#    plt.ylabel('Magnitude')
#    plt.title('Cluster Magnitude')
#    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_magnitude.png')
#    
#    # Plot magnitude vs cardinality
#    plt.scatter(cardinality, magnitudes)
#    plt.xlabel('Cardinality')
#    plt.ylabel('Magnitude')
#    plt.title('Cardinality x Magnitude')
#    plt.savefig(output_folder+'/'+experiment_name+'_kmeans_card_x_mag.png') 

    # Evaluate the clusters by using homogeneity, completeness and v score
    labels = em.predict(data)
    h, c, v = homogeneity_completeness_v_measure(act_labels, labels)
    
    return em, keep_k, keep_meth, keep_init, keep_iter, keep_cov, h, c, v, final_time

def myPCA(data, act_labels, output_folder, experiment_name):
    # Get num_features
    num_features = data.shape[1]    
    
    # First let's see how much variance is explained by each component
    pca_full = PCA(random_state=13).fit(data)
    ev = pca_full.explained_variance_ratio_*100
    cum = np.cumsum(ev)
    
    # Graph pareto chart
    fig, (ax, ax3) = plt.subplots(1,2)
    ax.bar(np.arange(len(ev)), ev)
    ax.set_title('Pareto of Explained Variance for '+experiment_name)
    ax2 = ax.twinx()
    ax2.plot(cum, marker="D", ms=7, color='black')
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim([0,100])
    ax3.plot(ev)
    ax3.set_title('Scree Plot for '+experiment_name)
    ax3.yaxis.set_major_formatter(PercentFormatter())
    fig.suptitle('Explained Variance per Component')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_folder+'/'+experiment_name+'_pca_pareto.png', bbox_inches='tight')
    plt.close()
    
    # Rerun PCA this time only keeping the top 2 components
    pca = PCA(n_components = 2, random_state=13).fit(data)
    ev = pca.explained_variance_ratio_*100
    cum = np.cumsum(ev)
    trans_data = pca.transform(data)
    
    # Graph the top 2 components 
    plt.scatter(trans_data[:,0], trans_data[:,1], c=act_labels)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Top 2 Principal Components')
    plt.savefig(output_folder+'/'+experiment_name+'_pca_top_2.png', bbox_inches='tight')
    plt.close()
    plt.figure()
    
    # Graph the reconstruction error
    mses = []
    for k in range(num_features):
        pca = PCA(n_components = k, random_state = 13).fit(data)
        trans_data = pca.transform(data)
        rec_data = pca.inverse_transform(trans_data)
        mse = MSE(rec_data, data.values)
        mses.append(mse)
    
    plt.plot(mses)
    plt.xticks(ticks=list(range(num_features)), labels=list(range(1, num_features+1)))
    plt.xlabel('# Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Average Reconstruction Error for K Components')
    plt.savefig(output_folder+'/'+experiment_name+'_pca_component_reconstruction_error.png')
    plt.close()
    plt.figure()
    
    # Finally run PCA taking the components that explain at least 80% of the variance
    start_time = time.time()
    pca = PCA(n_components=0.8, random_state=13).fit(data)
    end_time = time.time()
    final_time = end_time - start_time
    
    return pca, pca_full, final_time

def myICA(data, act_labels, output_folder, experiment_name):
    # Create random seed list
    rn = np.random.RandomState(13)
    random_seeds = list(rn.randint(1,1000000,1))
    
    # Vary the # of components from 1 to num features and run FastICA
    # Also we will run this for multiple random seeds and plot each
    num_features = data.shape[1]
    num_features = 55
    all_kurts_parallel = []
    all_kurts_deflation = []
    for r in random_seeds:
        kurts_parallel = []
        kurts_deflation = []
        for i in range(1,num_features):
            ica_parallel = FastICA(n_components=i, random_state=r, algorithm = 'parallel').fit(data)
            trans_data = ica_parallel.transform(data)
            k = np.mean(np.abs(kurtosis(trans_data)))
            kurts_parallel.append(k)
            ica_deflation = FastICA(n_components=i, random_state=r, algorithm = 'deflation').fit(data)
            trans_data = ica_deflation.transform(data)
            k = np.mean(np.abs(kurtosis(trans_data)))
            kurts_deflation.append(k)
            
        all_kurts_parallel.append(kurts_parallel)
        all_kurts_deflation.append(kurts_deflation)
        
    # Plot the kurtosis values agains number of components
    fig, (ax1, ax2) = plt.subplots(2)
    for kurt in all_kurts_parallel:
       ax1.plot(kurt)
    ax1.set_xticks(list(range(num_features-1)))
    ax1.set_xticklabels(list(range(1,num_features)))
    ax1.set_ylabel('Average Kurtosis')
    ax1.set_xlabel('Number of Components')
    ax1.set_title('Parallel Algorithm')
    for kurt in all_kurts_deflation:
        ax2.plot(kurt)
    ax2.set_xticks(list(range(num_features-1)))
    ax2.set_xticklabels(list(range(1,num_features)))
    ax2.set_ylabel('Average Kurtosis')
    ax2.set_xlabel('Number of Components')
    ax2.set_title('Deflation Algorithm')
    fig.suptitle('Average Kurtosis per Components')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_folder+'/'+experiment_name+'_ica_kurtosis.png', bbox_inches='tight')
    plt.close()
    
    num_features = data.shape[1]+1
    all_kurts_logcosh = []
    all_kurts_exp = []
    all_kurts_cube = []
    for r in random_seeds:
        kurts_logcosh = []
        kurts_exp = []
        kurts_cube = []
        for i in range(1,num_features):
            ica_logcosh = FastICA(n_components=i, random_state=r, fun='logcosh').fit(data)
            trans_data = ica_logcosh.transform(data)
            k = np.mean(np.abs(kurtosis(trans_data)))
            kurts_logcosh.append(k)
            ica_exp = FastICA(n_components=i, random_state=r, fun='exp').fit(data)
            trans_data = ica_exp.transform(data)
            k = np.mean(np.abs(kurtosis(trans_data)))
            kurts_exp.append(k)
            ica_cube = FastICA(n_components=i, random_state=r, fun='exp').fit(data)
            trans_data = ica_cube.transform(data)
            k = np.mean(np.abs(kurtosis(trans_data)))
            kurts_cube.append(k)
            
        all_kurts_logcosh.append(kurts_logcosh)
        all_kurts_exp.append(kurts_exp)
        all_kurts_cube.append(kurts_cube)
        
    # Plot the kurtosis values agains number of components
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,10))
    for kurt in all_kurts_logcosh:
       ax1.plot(kurt)
    ax1.set_xticks(list(range(num_features-1)))
    ax1.set_xticklabels(list(range(1,num_features)))
    ax1.set_ylabel('Average Kurtosis')
    ax1.set_xlabel('Number of Components')
    ax1.set_title('Log Cosh')
    for kurt in all_kurts_exp:
        ax2.plot(kurt)
    ax2.set_xticks(list(range(num_features-1)))
    ax2.set_xticklabels(list(range(1,num_features)))
    ax2.set_ylabel('Average Kurtosis')
    ax2.set_xlabel('Number of Components')
    ax2.set_title('EXP')
    for kurt in all_kurts_cube:
        ax3.plot(kurt)
    ax3.set_xticks(list(range(num_features-1)))
    ax3.set_xticklabels(list(range(1,num_features)))
    ax3.set_ylabel('Average Kurtosis')
    ax3.set_xlabel('Number of Components')
    ax3.set_title('Cube')
    fig.suptitle('Average Kurtosis per Components')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_folder+'/'+experiment_name+'_ica_kurtosis_2.png', bbox_inches='tight')
    plt.close()
    
    # Run ICA one more time with best hyper-parameters
    # Choose between deflation or parallel algorithm
    algorithms = ['deflation', 'parallel']
    defl = np.mean(np.max(np.array(all_kurts_deflation),axis=1))
    par = np.mean(np.max(np.array(all_kurts_parallel), axis=1))
    a = algorithms[np.argmax([defl, par])]
    # Choose between logcosh, exp, cube
    funcs = ['logcosh', 'exp', 'cube']
    lc = np.mean(np.max(np.array(all_kurts_logcosh), axis=1))
    exp = np.mean(np.max(np.array(all_kurts_exp),axis=1))
    cub = np.mean(np.max(np.array(all_kurts_cube), axis=1))
    f = funcs[np.argmax([lc, exp, cub])]
    # Figure out best num components
    votes = []
    votes.extend(list(np.argmax(np.array(all_kurts_deflation),axis=1)))
    votes.extend(list(np.argmax(np.array(all_kurts_parallel),axis=1)))
    votes.extend(list(np.argmax(np.array(all_kurts_logcosh),axis=1)))
    votes.extend(list(np.argmax(np.array(all_kurts_exp),axis=1)))
    votes.extend(list(np.argmax(np.array(all_kurts_cube),axis=1)))
    i = mode(votes)[0][0]+1 # Add one to account for python 0 indexing
    
    start_time = time.time()
    ica = FastICA(n_components=i, random_state=13, fun=f, algorithm=a).fit(data)
    end_time = time.time()
    final_time = end_time - start_time
    
    return ica, final_time

def myRCA(data, act_labels, output_folder, experiment_name):
    # Let's start by exploring the random projections we get for different number of components
    num_features = data.shape[1]
    values = list(range(1, num_features))
    rn = np.random.RandomState(13)
    random_seeds = list(rn.randint(1,1000000,20))
    errors = []
    for r in random_seeds:
        mses = []
        for k in values:
            rca = GaussianRandomProjection(n_components = k, random_state = r).fit(data)
            trans_data = rca.transform(data)
            inv_data = np.linalg.pinv(rca.components_.T)
            rec_data = trans_data.dot(inv_data)
            mse = MSE(rec_data, data.values)
            mses.append(mse)
        errors.append(mses)
    avg_errors = np.mean(np.array(errors),axis=0)
    std_errors = np.std(np.array(errors),axis=0)
    
    # Graph the reconstruction error per component
    plt.errorbar(list(range(1, num_features)), avg_errors, std_errors)
    plt.xticks(ticks=list(range(num_features)), labels=list(range(1, num_features+1)))
    plt.xlabel('# Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Average Reconstruction Error for K Components Over 200 Iterations')
    plt.savefig(output_folder+'/'+experiment_name+'_rca_component_reconstruction_error.png')
    plt.close()
    plt.figure()
    
    # Create a final rca to return
    k = np.argmin(avg_errors)+1 # add 1 to account for 0 indexing
    thresh = 0.2
    for i in range(len(avg_errors)):
        if avg_errors[i] <= thresh:
            k = i+1 # Add 1 to account for 0 indexing
            break
        
    start_time = time.time()
    rca = GaussianRandomProjection(n_components = k, random_state=13).fit(data)
    end_time = time.time()
    final_time = end_time - start_time
    
    return rca, final_time

def myFS(data, act_labels, output_folder, experiment_name, avg='binary'):
    
    # Split data, act_labels into train, test sets
    X_train, X_test, y_train, y_test = train_test_split(data, act_labels, test_size=0.1, random_state=13)
    
    # Calculate test recall when running multiple iterations for k = 1 to num_features
    num_features = data.shape[1]
    num_features = 25
    values = list(range(1, num_features+1))
    rn = np.random.RandomState(13)
    random_seeds = list(rn.randint(1,1000000, 1))
    recalls = []
    for r in random_seeds:
        recall_temp = []
        for k in values:
            estimator = DecisionTreeClassifier(random_state=r)
            fs = RFE(estimator, n_features_to_select=k, step=1).fit(X_train, y_train)
            y_pred = fs.predict(X_test)
            rec = recall_score(y_test, y_pred, average = avg)
            recall_temp.append(rec)
        recalls.append(recall_temp)
    
    avg_recall = np.mean(np.array(recalls),axis=0)
    recall_std = np.std(np.array(recalls), axis=0)
    
    # Plot the average recall for each k and include the error bars
    plt.errorbar(list(range(1, num_features+1)), avg_recall, recall_std)
    plt.xticks(ticks=list(range(1, num_features+1)), labels=list(range(1, num_features+1)))
    plt.xlabel('# Components')
    plt.ylabel('Recall Score')
    plt.title('Average Recall Score for K Components Over 20 Iterations')
    plt.savefig(output_folder+'/'+experiment_name+'_fs_component_recall_score.png')
    plt.close()
    plt.figure()
    
    # Plot the reconstruction errors
    errors = []
    for r in random_seeds:
        mses = []
        for k in values:
            estimator = DecisionTreeClassifier(random_state=r)
            fs = RFE(estimator, n_features_to_select=k, step=1).fit(X_train, y_train)
            trans_data = fs.transform(X_train)
            rec_data = fs.inverse_transform(trans_data)
            mse = MSE(rec_data, X_train.values)
            mses.append(mse)
            errors.append(mses)
    avg_errors = np.mean(np.array(errors),axis=0)
    std_errors = np.std(np.array(errors),axis=0)
    
    plt.errorbar(list(range(1, num_features+1)), avg_errors, std_errors)
    plt.xticks(ticks=list(range(num_features)), labels=list(range(1, num_features+1)))
    plt.xlabel('# Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Average Reconstruction Error for K Components')
    plt.savefig(output_folder+'/'+experiment_name+'_fs_component_reconstruction_error.png')
    plt.close()
    plt.figure()
    
    # Run one more time with the optimal k value found above
    votes = [np.argmax(avg_recall), np.argmax(avg_recall+recall_std), np.argmax(avg_recall-recall_std)]
    results = np.zeros(num_features)
    for v in votes:
        results[v] = results[v]+1
    k = np.argmax(results)  
    estimator = DecisionTreeClassifier(random_state=13)
    start_time = time.time()
    fs = RFE(estimator, n_features_to_select=k, step=1).fit(X_train, y_train)
    end_time = time.time()
    final_time = end_time - start_time
    
    return fs, final_time

def churnNN_tuning(training_set):
    # Set random seeds
    torch.manual_seed(13)
    np.random.seed(13)
    
    # Calculate number of features
    nf = training_set[0].shape[1]
    
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
        
    # Initialize best hyper-parameters
    best_alpha = 0.01
    best_gamma = 0.9
    best_recall = 0.0
    
    # Lists of alpha-gamma pairs to try
    alphas = [0.01, 0.001, 0.0001]
    gammas = [0.9, 0.7, 0.5, 0.3, 0.1]
    
    # Define X,y
    x,Y = training_set
    
    # Create KFolds to do cross validation
    kf = KFold(n_splits=3)
    kf.get_n_splits(x)
    
    for alpha in alphas:
        for gamma in gammas:
            recalls = []
            for train_index, test_index in kf.split(x):
                X, X_test = x[train_index], x[test_index]
                y, y_test = Y[train_index], Y[test_index]
                
                model = MLP(nf,10,7,5)
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr = alpha, momentum=gamma)
                
                x_train = torch.FloatTensor(X)
                Y_train = torch.FloatTensor(y)
                x_test = torch.FloatTensor(X_test)
                Y_test = torch.FloatTensor(y_test)
                model.train()
                epoch = 10000
                
                training_loss = []
                for epoch in range(epoch):
                    optimizer.zero_grad()
                    y_insample = model(x_train)
                    loss = criterion(y_insample.squeeze(), Y_train)
                    training_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
            
                model.eval()
                y_outsample = model(x_test)
                after_train = criterion(y_outsample.squeeze(), Y_test)
                Y_test = Y_test.detach().numpy().astype(int)
                y_outsample = np.rint(y_outsample.detach().numpy().flatten())
                out_recall=recall_score(Y_test,y_outsample)
                recalls.append(out_recall)
            recall = np.mean(recalls)
            if recall > best_recall:
                best_alpha = alpha
                best_gamma = gamma
    
    return best_alpha, best_gamma


def churnNN(training_set, X_test, y_test, output_folder, experiment_name, alpha=0.01, gamma=0.9):
    # Set random seeds
    torch.manual_seed(13)
    np.random.seed(13)
    
    # Calculate number of features
    nf = training_set[0].shape[1]
    
    # Find X, y
    X,y = training_set
    
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
    
    start_time = time.time()
    model = MLP(nf,10,7,5)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = alpha, momentum=gamma)
    
    x_train = torch.FloatTensor(X)
    Y_train = torch.FloatTensor(y)
    x_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(y_test)
    model.train()
    epoch = 10000
    
    training_loss = []
    for epoch in range(epoch):
        optimizer.zero_grad()
        y_insample = model(x_train)
        loss = criterion(y_insample.squeeze(), Y_train)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time

    model.eval()
    start_time = time.time()
    y_insample = model(x_train)
    end_time = time.time()
    in_query_time=end_time-start_time
    Y_train = Y_train.detach().numpy().astype(int)
    y_insample = np.rint(y_insample.detach().numpy().flatten())
    in_recall=recall_score(Y_train,y_insample)
    
    start_time = time.time()
    y_outsample = model(x_test)
    end_time = time.time()
    out_query_time=end_time-start_time
    after_train = criterion(y_outsample.squeeze(), Y_test)
    Y_test = Y_test.detach().numpy().astype(int)
    y_outsample = np.rint(y_outsample.detach().numpy().flatten())
    out_recall=recall_score(Y_test,y_outsample)

    return training_loss, in_recall, out_recall, training_time, in_query_time, out_query_time

def NN_plots(training_loss_all, in_recalls, out_recalls, training_times, output_folder, experiment_name):
    # Plot Loss curves for different size training sets
    legend = []
    for y_vals in training_loss_all:
        ln, = plt.plot(y_vals)
        legend.append(ln)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Comparison of Training Loss\nPer Epoch for Neural Network')
    plt.legend(legend, ['N=100', 'N=1000', 'N=2500', 'N=5000', 'N=8000'], title='Training Size')
    plt.savefig(output_folder+'/'+experiment_name+'_NN_training_loss.png')
    plt.close()
    
    # Plot recall for each training set
    fig, (ax1, ax2) = plt.subplots(1, 2)
#    fig.suptitle('Recall Performance by Training Size')
    colors = ['#191970','#00008B','#0000CD','#0000FF','#4169E1']
    ax1.barh(np.arange(len(in_recalls)), in_recalls, color=colors)
    ax1.set_yticks(ticks=list(range(len(in_recalls))))
    ax1.set_yticklabels(['N=100', 'N=1000', 'N=2500', 'N=5000', 'N=8000'])
    ax1.set_title('In Sample Recall Score')
    ax2.barh(np.arange(len(out_recalls)), out_recalls, color=colors)
    ax2.set_yticks(ticks=list(range(len(out_recalls))))
    ax2.set_yticklabels(['N=100', 'N=1000', 'N=2500', 'N=5000', 'N=8000'])
    ax2.set_title('Out of Sample Recall Score')
#    fig.suptitle('Recall Score Comparison')
    plt.tight_layout()
    fig.savefig(output_folder+'/'+experiment_name+'_NN_recall_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Plot training times for each training set
    plt.barh(np.arange(len(training_times)), training_times, color=colors)
    plt.yticks(ticks=list(range(len(['N=100', 'N=1000', 'N=2500', 'N=5000', 'N=8000']))), labels=['N=100', 'N=1000', 'N=2500', 'N=5000', 'N=8000'])
    plt.xlabel('Training Time')
    plt.ylabel('Training Size')
    plt.title('Trainig Time (Wall Clock)')
    plt.tight_layout()
    plt.savefig(output_folder+'/'+experiment_name+'_NN_wall_clock.png',bbox_inches='tight')
    
def NN_comparison_plot(final_recalls, final_times, algos, output_folder, experiment_name):
    # Plot recall for each training set
    colors = ['gray', 'red', 'orange', 'green', 'blue']
    plt.barh(np.arange(len(final_recalls)), final_recalls, color=colors)
    plt.title('Recall Performance by Algorithm')
    plt.yticks(ticks=list(range(len(final_recalls))), labels=algos)
    plt.tight_layout()
    plt.savefig(output_folder+'/'+experiment_name+'_NN_recall_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Plot training times for each model
    colors = ['gray', 'red', 'orange', 'green', 'blue']
    plt.barh(np.arange(len(final_times)), final_times, color=colors)
    plt.yticks(ticks=list(range(len(algos))), labels=algos)
    plt.xlabel('Training Time')
    plt.ylabel('Model')
    plt.title('Trainig Time (Wall Clock)')
    plt.tight_layout()
    plt.savefig(output_folder+'/'+experiment_name+'_NN_wall_clock_comparison.png',bbox_inches='tight')