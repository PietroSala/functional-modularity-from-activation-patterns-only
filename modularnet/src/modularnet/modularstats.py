from modularnet.modularcluster import ModularClusterDBScan, ModularClusterHDBScan, ModularClusterKmeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import math
import numpy as np
import matplotlib.pyplot as plt 
from modularnet.modularspace import ModularSpace
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from utils.utils import session_path

class ModularStats:
    def __init__(self, space: ModularSpace, metric_covariance=False, metric_count=False, base_path=None):
        if base_path is None: base_path = session_path()
        self.base_path = base_path
        self.space = space
        self._metric_covariance = metric_covariance
        self._metric_count = metric_count

        self.metric_covariance(metric_covariance)
        self.metric_count(metric_count)

    def stats(self):
        mos = self.space.mos
        positions_count = [mo.stats_positions_count for mo in mos]
        data_count = [mo.stats_data_count for mo in mos]
        
        for pos, cnt in zip(positions_count, data_count):
            perc = ((pos/cnt)*100)
            q75 = torch.quantile(perc, 0.75).type(torch.int).tolist()
            q50 = torch.quantile(perc, 0.50).type(torch.int).tolist()
            q25 = torch.quantile(perc, 0.25).type(torch.int).tolist()
            perc = perc.type(torch.int).tolist()
            print(q75,q50,q25)
            print(perc)

        dist_on_list = [np.array(mo.stats_dist_on) for mo in mos]
        dist_center_list = [np.array(mo.stats_dist_center) for mo in mos]
        layers = len(dist_on_list)
        fig, axs = plt.subplots(1,layers)
        for i,(dist_on, dist_center, ax) in enumerate(zip(dist_on_list, dist_center_list, axs)):
            ax.set_title("Layer "+str(i))
            ax.plot(dist_on, label='dist_on', color='green')
            ax.plot(dist_center, label='dist_center', color='red')
            ax.legend()
        plt.show(block=False)

    
    
    def metric_count(self, value:bool):
        for mo in self.space.mos:
            mo.metric_count = value
            mo.metric_count_total = 0

    def show_metric_count(self):
        plt.clf()
        cnt = len(self.space.mos)
        cols = max(math.floor(math.sqrt(cnt)),1)
        rows = max( math.ceil(cnt / cols),1)
        plt.suptitle(f"Metric Count")
        for i, mo in enumerate(self.space.mos):
            cnts = torch.zeros(mo.input_size,mo.input_size)
            for k,v in mo.counts.items():
                cnts[k] = v
                cnts[(k[1],k[0])] = v
            cnts_rel = cnts / mo.metric_count_total
            score_made = self.calculate_score_made(cnts_rel)
            pol_score = self.polarization_score(cnts_rel)

            ax = plt.subplot(rows,cols,i+1)
            ax.imshow(cnts_rel.numpy(),cmap='hot', interpolation='nearest')
            plt.title(f"Layer{i} - made: {score_made:.5f} - pol: {pol_score:.5f}")
        
        mod = 'modular' if self.space.modularizing else 'vanilla'
        plt.savefig(f'{self.base_path}/metric_count_{mod}.png')
        plt.show(block=False)  
  

    def metric_covariance(self, value:bool):
        for mo in self.space.mos:
            mo.metric_covariance = value

    def show_metric_covariance(self):
        
        plt.clf()
        cnt = len(self.space.mos)
        cols = max(math.floor(math.sqrt(cnt)),1)
        rows = max(math.ceil(cnt / cols),1)
        plt.suptitle(f"Metric Covariance")
        for i, mo in enumerate(self.space.mos):
            if mo.metric_covariance == False: return
            all_activations = torch.concatenate(mo.activations)
            all_activations = all_activations.flatten(1)
            covariance_matrix = torch.cov(all_activations.T)

            score_made = self.calculate_score_made(covariance_matrix)
            pol_score = self.polarization_score(covariance_matrix)

            ax = plt.subplot(rows,cols,i+1)
            ax.imshow(covariance_matrix.cpu().numpy(),cmap='hot', interpolation='nearest')
            plt.title(f"Layer{i} - made: {score_made:.5f} - pol: {pol_score:.5f}")
        
        mod = 'modular' if self.space.modularizing else 'vanilla'
        
        plt.savefig(f'{self.base_path}/metric_cov_{mod}.png')
        plt.show(block=False)  

    def calculate_score_made(self, matrix: torch.Tensor):
        # Calculate absolute distance from 0 and 1 simultaneously
        dist_from_0 = torch.abs(matrix)
        dist_from_1 = torch.abs(matrix - 1)
        
        # Get minimum distance to either extreme
        min_distances = torch.minimum(dist_from_0, dist_from_1)
        
        # Calculate mean across all elements
        made = torch.mean(min_distances).item()
        
        return made

    def polarization_score(self, covariance_matrix: torch.Tensor):
        flattened_matrix = covariance_matrix.flatten()
        distances = torch.abs(flattened_matrix - 0.5)
        mean_distance = torch.mean(distances)
        polarization_score = 1 / (mean_distance + 1e-10)  # Avoid division by zero

        return polarization_score
    
    def cluster_space_dbscan(self):
        plt.clf()
        cnt = len(self.space.mos)
        cols = max(math.floor(math.sqrt(cnt)),1)
        rows = max(math.ceil(cnt / cols),1)

        """
        plt.suptitle(f"Cluster DBSCAN")
        for i, mo in enumerate(self.space.mos):
            pos = mo.positions
            
            ax = plt.subplot(rows,cols,i+1)

            self.perform_dbscan_clustering(pos, ax)
        """

        plt.clf()
        plt.suptitle(f"Cluster DBSCAN all")
        all_positions = torch.cat([mo.positions for mo in self.space.mos])
        
        #for i, mo in enumerate(self.space.mos):
        #dists = torch.cdist(all_positions,all_positions)
        #eps = dists.max().item() # dists.mean().item()
        #min_samples = max(len(all_positions)//10,2)
        
        self.perform_dbscan_clustering(all_positions)

    def perform_dbscan_clustering(self, data, ax=plt, normalize=False):
        
        if normalize:
            data = torch.functional.F.normalize(data, p=2, dim=1)
        
        data = data.detach().cpu().numpy()

        eps,min_sample = self.estimate_dbscan_eps_dist(data)

        #pca = PCA(n_components=2)
        #data_2d = pca.fit_transform(data)

        dbscan = DBSCAN(eps=eps, min_samples=min_sample)
        labels = dbscan.fit_predict(data)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        metric_sil = silhouette_score(data, labels) if n_clusters > 1 else -1

        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        ax.scatter(data_2d[:,0], data_2d[:,1], c=labels, s=20, cmap='viridis')
        plt.title(f"Layers - Clusters: {n_clusters} - Silhouette: {metric_sil:.5f}")
        
        mod = 'modular' if self.space.modularizing else 'vanilla'
        plt.savefig(f'{self.base_path}/metric_cluster_all_dbscan_{mod}.png')
        plt.show(block=True)

    # Step 1: Automatically estimate `eps` using the k-distance graph
    def estimate_dbscan_eps_dist(self, data, k=5):
        # Compute the k-nearest neighbors distances
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        
        # Sort the k-th nearest neighbor distances
        k_distances = np.sort(distances[:, k-1], axis=0)
        
        # Find the elbow point using the KneeLocator
        kneedle = KneeLocator(np.arange(len(k_distances)), k_distances, curve='convex', direction='increasing')
        eps = k_distances[kneedle.elbow] if kneedle.elbow else np.percentile(k_distances, 95)  # Fallback to 95th percentile
        min_samples = 2 * data.shape[1]
        print('eps', eps, 'min_samples',min_samples)
        return eps, min_samples


    def estimate_dbscan_params(self, X, k_dist=4, plot=False):
        """
        Estimate DBSCAN parameters (eps and min_samples) for a given dataset.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data
        k_dist : int, default=4
            The k value for k-distance graph (usually between 2-5)
        plot : bool, default=True
            Whether to show the k-distance plot
        
        Returns:
        --------
        eps : float
            Estimated eps parameter
        min_samples : int
            Estimated min_samples parameter
        """
        # Calculate distances to k nearest neighbors for each point
        neigh = NearestNeighbors(n_neighbors=k_dist)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        
        # Sort distances to kth nearest neighbor
        k_dist = np.sort(distances[:, -1])
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(k_dist)), k_dist)
            plt.xlabel('Points sorted by distance')
            plt.ylabel(f'Distance to {k_dist}th nearest neighbor')
            plt.title('K-distance Graph')
            plt.grid(True)
            plt.show()
        
        # Find the elbow point using the maximum curvature
        coords = np.vstack((range(len(k_dist)), k_dist)).T
        
        # Line from first to last point
        line_vec = coords[-1] - coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        # Vector from point to first point
        vec_from_first = coords - coords[0]
        
        # Distance to line
        dist_from_line = np.cross(line_vec_norm, vec_from_first)
        
        # Find the point with max distance (elbow)
        elbow_ind = np.argmax(np.abs(dist_from_line))
        eps = k_dist[elbow_ind]
        
        # Estimate min_samples as k_dist
        min_samples = k_dist
        
        return eps, elbow_ind # min_samples