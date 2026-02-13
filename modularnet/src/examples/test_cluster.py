import math
import torch
from sklearn.cluster import KMeans, DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import umap
import itertools


basepath = '/home/cesare/Projects/StrangeNet/checkpoint/'

def main():
    positions = torch.load(basepath + 'fashion_mnist_cnn_space.pth', weights_only=True)
    positions_all = torch.concatenate(positions, dim=0)
    
    print(positions_all.shape)
    #print(positions.shape)
    positions_all = positions_all.detach().cpu().numpy()


    ## KMEANS + elbow
    lbl_kmeans = cluster_kmeans(positions_all)
    scores_kmeans = cluster_scores(positions_all, lbl_kmeans)
    plot_cluster(positions_all, lbl_kmeans, "KMEANS")

    #num_samples = np.unique_counts(lbl_kmeans)
    #Smin_num_samples = int( (np.min(num_samples) + np.median(num_samples))// 2)

    ## DBSCAN + esitmated eps, min_samples
    lbl_dbs = cluster_dbscan(positions_all) #, min_num_samples)
    scores_dbs = cluster_scores(positions_all, lbl_dbs)
    plot_cluster(positions_all, lbl_dbs, "DBScan")
    

    ## HDBSCAN
    lbl_hdbs = cluster_hdbscan(positions_all) #, min_num_samples)
    scores_hdbs = cluster_scores(positions_all, lbl_hdbs)
    plot_cluster(positions_all, lbl_hdbs, "HDBScan")

    header = [' '*7,'Silhouette ↑', 'Calinski-Harabasz ↑', 'Davies-Bouldin ↓']
    scores = [scores_kmeans, scores_dbs, scores_hdbs]
    names = ['KMeans', 'DBSCAN', 'HDBSCAN']

    print(header)
    for score, name in zip(scores, names):
        print(name, score)

def plot_elbow(ks, inertias, elbow, knee):
    print(knee, elbow)
    plt.plot(ks, inertias)
    plt.suptitle(f'{knee} {elbow}')
    plt.show()
    

def cluster_kmeans_elbow(data, plot = True):
    ks = list(range(1, 30))
    inertias = []
    for k in ks:
        kmean = KMeans(n_clusters=k)
        kmean.fit(data)
        inertias.append(kmean.inertia_)
    kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
    if plot: plot_elbow(ks, inertias, kn.knee, kn.elbow)
    best = round((kn.knee + kn.elbow) / 2)

    return best

# hyper param search: best silhouette_score

def cluster_kmeans_search_score(data, plot = True):
    ks = list(range(1, 50))
    n_clusters = []
    scores = []
    for k in ks:
        kmean = KMeans(n_clusters=k)
        labels = kmean.fit_predict(data)
        num = len(np.unique(labels[labels != -1]))
        score = -1 if num <= 1 else silhouette_score(data, labels)
        
        scores.append(score)

    best_idx = np.argmax(scores)
    best_k = ks[best_idx]

    return best_k

def cluster_dbscan_search_score(data, plot = True):
    cdist = pairwise_distances(data)
    eps_min, eps_max = cdist.min(), cdist.max()
    epss = np.linspace(eps_min + 0.001, eps_max / 10, num=20).tolist()
    n_data = len(data)
    n_samples = list(range(n_data//50,n_data//20)) 
    
    scores = []
    params = []
    for n_sample in n_samples:
        for eps in epss:
            dbs = DBSCAN(min_samples=n_sample, eps=eps)
            labels = dbs.fit_predict(data)
            num = len(np.unique(labels[labels != -1]))
            score = -1 if num <= 1 else silhouette_score(data, labels)

            params.append((n_sample, eps))
            scores.append(score)
       
    
    # Find the best epsilon (max Silhouette Score)
    best_idx = np.argmax(scores)
    best_samples, best_eps = params[best_idx]

    return best_samples, best_eps 

def cluster_hdbscan_search_score(data, plot = True):
    cdist = pairwise_distances(data)
    eps_min, eps_max = cdist.min(), cdist.max()
    epss = np.linspace(eps_min + 0.001, eps_max / 10, num=20).tolist()
    n_data = len(data)
    n_samples = list(range(n_data//50,n_data//20))
    
    scores = []
    params = []
    for n_sample in n_samples:
        for eps in epss:
            kmean = HDBSCAN(min_samples=n_sample, cluster_selection_epsilon=eps)
            labels = kmean.fit_predict(data)
            num = len(np.unique(labels[labels != -1]))
            score = -1 if num <= 1 else silhouette_score(data, labels)
            
            params.append((n_sample, eps))
            scores.append(score)

    best_idx = np.argmax(scores)
    best_samples, best_eps = params[best_idx]

    return best_samples, best_eps

# perform the custering after hyperparam search 

def cluster_kmeans(data, k=None):
    if k is None:
        k = cluster_kmeans_search_score(data, plot=False)
    kmean = KMeans(n_clusters=k,)
    labels = kmean.fit_predict(data)
    unique_labels = np.unique_counts(labels)
    print(unique_labels)
    return labels


def cluster_dbscan(data, min_samples=None, eps=None):
    if min_samples is None or eps is None:
        min_samples, eps  = cluster_dbscan_search_score(data, plot=False)
    dbs = DBSCAN(min_samples=min_samples,eps=eps)
    labels = dbs.fit_predict(data)
    unique_labels = np.unique_counts(labels)
    print(unique_labels)
    return labels

def cluster_hdbscan(data, min_samples=None, eps=None):
    if min_samples is None or eps is None:
        min_samples, eps  = cluster_hdbscan_search_score(data, plot=False)
    hdbs = HDBSCAN(min_samples, cluster_selection_epsilon=eps)
    labels = hdbs.fit_predict(data)
    unique_labels = np.unique_counts(labels)
    print(unique_labels)
    return labels



## dim reduction
    
    
def reduce_TSNE(data):
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(data)
    return data_tsne

def reduce_PCA(data):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    return data_pca

def reduce_UMAP(data):
    umap_reducer = umap.UMAP(n_components=2)
    data_umap = umap_reducer.fit_transform(data)
    return data_umap

#plots

def plot_cluster(data, labels, title):
    data_vis = [data]
    data_txt = ['RAW']

    if data.shape[1] > 2:
        data_tsne = reduce_TSNE(data)
        data_pca = reduce_PCA(data)
        data_umap = reduce_PCA(data)
        data_vis = [data_tsne, data_pca, data_umap]
        data_txt = ['TSNE', 'PCA', 'UMAP']
        

    cnt = len(data_vis)
    rows, cols = plot_row_cols(cnt)
    plt.suptitle(title)
    for i, (data,txt) in enumerate( zip(data_vis, data_txt) ):
        ax = plt.subplot(rows, cols, i+1)
        ax.set_title(txt)
        ax.scatter(data[:,0], data[:,1], c=labels )
    plt.show()

def plot_row_cols(cnt):
    rows = max(1,math.ceil(cnt*0.5))
    cols = math.ceil(cnt/rows)
    return rows, cols

#metrics 

def cluster_scores(data, labels):
    #score_ari = adjusted_rand_score(labels, labels)
    score_silhouette = silhouette_score(data, labels)
    score_calinski = calinski_harabasz_score(data, labels)
    score_davies = davies_bouldin_score(data, labels)

    #print(f"ARI: {score_ari}")
    #print(f"Silhouette: {score_silhouette}")
    #print(f"Calinski-Harabasz: {score_calinski}")
    #print(f"Davies-Bouldin: {score_davies}")

    #return score_ari, score_silhouette, score_calinski, score_davies
    return float(score_silhouette), float(score_calinski), float(score_davies)
    

if __name__ == '__main__': main()