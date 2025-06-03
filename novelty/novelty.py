import ast
from collections import Counter
import re
import ast

import json
from tqdm import tqdm
from math import log
import heapq

from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
import nltk
from collections import OrderedDict


from novelty.divergences import Jensen_Shannon
import numpy as np
import random
import importlib
# importlib.reload(divergences)
from scipy.sparse import csr_matrix
from scipy.special import rel_entr
from heapq import heappush, heappop
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import parallel_backend, Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
import math



class Newness():
    """
    Estimate the novelty of a distribution by comparing it to a known reference distribution.

    This class computes the proportion of new terms in a distribution, as well as the proportion of disappearing terms. 

    Parameters:
        known_P (array-like): The known/reference probability distribution (e.g., KS and ES).
        new_Q (array-like): The new or current probability distribution to compare against `known_P`.
        lambda_ (float, optional): Weighting factor (default is 0.8). 

    Attributes:
        known_P (array-like): The known distribution.
        new_Q (array-like): The new distribution.
        lambda_ (float): The weighting factor.
        JSD_vector (array-like): Jensen-Shannon divergence vector between `known_P` and `new_Q`, calculated component-wise.
        nb_elements (int): The number of elements in the JSD vector.

    Notes:
        - The two approaches to estimating novelty (new vs. disappearing terms) are mathematically equivalent
          if thresholds are adjusted accordingly.
        - Choose the perspective (new or disappearing terms) based on interpretability needs.
    """
    def __init__(self, known_P, new_Q, lambda_=0.8):
        """
        Initialize the Newness object.

        Parameters:
            known_P (array-like): The reference probability distribution.
            new_Q (array-like): The distribution to be compared for novelty.
            lambda_ (float, optional): Weighting factor for further use (default = 0.8).
        """
        self.known_P = known_P
        self.new_Q = new_Q
        self.lambda_ = lambda_
        
        JS = Jensen_Shannon()
        self.JSD_vector = JS.linear_JSD(known_P, new_Q)
        self.nb_elements = len(self.JSD_vector)

    def divergent_terms(self, thr_div=0.0041, thr_new=0.5):

        """
        Identify terms with Jensen-Shannon divergence above a threshold and compute their newness ratio.
        
        Returns weighted ratios of appearing and and flags novelty if above `thr_new`.
        
        Parameters:
            thr_div (float): Default 0.0041, could be changed for each IPC class & year
            thr_new (float): Newness threshold to flag novelty.
        
        Returns:
            tuple: (newness score, novelty flag)

        Notes:
            JSD == 0 if and only if pi = qi, but we want to make sure the distribution gap between this two are large enough
            To interpret as if the new term make the divergence greater than threshold, then it is a significant cointributing, we just need to know in appearing or disappearing
        
        """
        count_appear = 0
        count_disappear = 0
        for i in range(self.nb_elements):
            if self.JSD_vector[i] > thr_div:
                if self.new_Q[i] > self.known_P[i]:
                    count_appear += 1
                if self.known_P[i] > self.new_Q[i]:
                    count_disappear += 1

        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1-self.lambda_) * disappear_ratio
        novelty = 0
        if newness > thr_new:
            novelty = 1
        
        return newness, novelty

    def probable_terms(self, thr_prob=2, cte = 1e-10, thr_new=0.5):
        """
        Identify terms with significant probability ratio changes and compute their newness ratio.
        
        Returns weighted ratios of appearing and disappearing terms and flags novelty if above `thr_new`.
        
        Parameters:
            thr_prob (float): Probability ratio threshold to identify significant terms.
            cte (float): Small constant to avoid division by zero.
            thr_new (float): Newness threshold to flag novelty.
        
        Returns:
            tuple: (newness score, novelty flag)
        Note:
            To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
        """
        count_appear = 0
        count_disappear = 0
        for i in range(self.nb_elements):
            if self.JSD_vector[i] != 0:
                # To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
                if self.new_Q[i] / (self.known_P[i]+cte) > thr_prob:  
                    count_appear += 1
                if self.known_P[i] / (self.new_Q[i]+cte) > thr_prob:
                    count_disappear += 1
        
        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1 - self.lambda_) * disappear_ratio
        novelty = 0
        if newness > thr_new:
            novelty = 1
        
        return newness, novelty


class Uniqueness():
    """
    Estimate the divergence of a new distribution from a reference (prototype) distribution.
    
    Compares term distributions to assess how much a new document or group deviates from the global norm.
    
    Parameters:
        known_P (array-like): Reference probability distribution (e.g., overall corpus distribution (KS or ES)).
    """
    def __init__(self, known_P):
        self.known_P = known_P
        #self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        
    def dist_to_proto(self, new_Q, thr_uniq=0.05):
        """
        Compute Jensen-Shannon divergence between a new distribution and the prototype.
        
        Flags uniqueness if divergence exceeds `thr_uniq`.

        Parameters:
            new_Q (array-like): Probability distribution of a new document or group.
            thr_uniq (float): Threshold to consider the distribution unique.
        
        Returns:
            tuple: (uniqueness score, uniqueness flag)
        """
        novel_uniq = 0
        uniqueness_ = self.JS.JSDiv(self.known_P, new_Q)
        if uniqueness_ > thr_uniq:
            novel_uniq = 1
            
        return uniqueness_, novel_uniq

    def proto_dist_shift(self, new_P, thr_uniqp=0.05):
        """
        Assess shift in the prototype distribution after incorporating new data.
        
        Flags significant shift if divergence exceeds `thr_uniqp`.

        Parameters:
            new_P (array-like): Updated or combined probability distribution.
            thr_uniqp (float): Threshold to flag distribution shift.
        
        Returns:
            tuple: (shift score, shift flag)
        """
        #new_P = self.known_P + self.new_Q
        uniqueness = self.JS.JSDiv(self.known_P, new_P)
        novel_uniq = 0
        if uniqueness > thr_uniqp:
            novel_uniq = 1

        return uniqueness, novel_uniq

    
class Difference():
    """
    Compute distributional differences between a reference distribution (`new_Q`) and a set of known distributions (`list_know_P`).

    This class allows estimating how different `new_Q` is from existing data by computing average Jensen-Shannon (JS) distances
    and determining how many neighbors differ beyond a given threshold.

    Parameters:
        list_know_P : scipy.sparse matrix
            Matrix of known document distributions (one per row).
        new_Q : numpy.ndarray
            Distribution vector representing a new document or variation.
        N : int, default=5
            Number of neighbors to consider when computing neighbor-based distances.
    """
    def __init__(self, list_know_P, new_Q, N=5):

        self.list_know_P = list_know_P.tocsr()
        self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        self.N = N
       

    def dist_estimate(self, sample=False, sample_size=1000, do_sample_P=False):
        """
        Estimate the average Jensen-Shannon distance to the N closest neighbors.

        Parameters:
            sample : bool, default=True
                Whether to randomly sample a subset of rows from `list_know_P` for estimation.
            sample_size : int, default=1000
                Number of samples to draw from `list_know_P` if `sample` is True.
            do_sample_P : bool, default=True
                Whether to use a reduced sample of `list_know_P` (up to 10,000 entries) as neighbor candidates (to find N closest neighbors).

        Returns:
            avg_final : float
                Average JS distance to the N closest neighbors across all sampled points.
        """
        num_points = self.list_know_P.shape[0]
        
        # Ensure the matrix is in CSR format for efficient row access
        self.list_know_P = self.list_know_P.tocsr()

        # Randomly sample `sample_size` indices
        if sample:
            sampled_indices = random.sample(range(num_points), min(sample_size, num_points))
        else: 
            sampled_indices = list(range(num_points))
        print(len(sampled_indices))
        # Sample of list_know_P
        if do_sample_P:
            sample_P = random.sample(range(num_points), min(10000, num_points))
        else:
            sample_P = sampled_indices

        avg_dists = []  # Store average distances
        all_dists_per_point = []  # Store all distances for each point

        def compute_distance(j):
                P_j = self.list_know_P[j].toarray().flatten()
                return Jensen_Shannon().JSDiv(P_i, P_j)
  
        for i in tqdm(sampled_indices):
            P_i = self.list_know_P[i].toarray().flatten()  # Convert sparse row to dense array

            # Compute distances to all other points in parallel
            all_dists = Parallel(n_jobs=-1, batch_size=int(num_points / os.cpu_count()))(
                delayed(compute_distance)(j) for j in sample_P if j != i  
            )
            # Take the smallest N distances if there are enough distances
            if len(all_dists) > self.N:
                all_dists = heapq.nsmallest(self.N, all_dists)

            avg_dist_i = sum(all_dists) / len(all_dists)
            avg_dists.append(avg_dist_i)
            all_dists_per_point.append(all_dists)

        # Compute final average distance
        avg_final = sum(avg_dists) / len(avg_dists)
        return avg_final


    def ratio_to_all(self, neighbor_dist, thr_diff=0.95):
        """
        Compare the distance from `new_Q` to all known distributions and compute the proportion
        that are farther than a given distance threshold.

        Parameters:
            neighbor_dist : float
                Distance threshold used as a reference (e.g., average distance to nearest neighbors).
            thr_diff : float, default=0.95
                Proportion threshold. If the proportion of known points with distance > `neighbor_dist`
                exceeds this threshold, `new_Q` is considered novel.

        Returns:
            difference : float
                Proportion of known distributions with distance > `neighbor_dist`.
            novel_diff : int
                1 if `difference` > `thr_diff`, indicating novelty, otherwise 0.
            dists : list of float
                All computed distances from known distributions to `new_Q`.
        """
        count_diff = 0
        num_known_P = self.list_know_P.shape[0]
        dists =[]
        for i in range(num_known_P):
            P_i = self.list_know_P[i].toarray().flatten()
            distance = self.JS.JSDiv(P_i, self.new_Q)
            dists.append(distance)
            if distance > neighbor_dist:
                count_diff += 1

        # Compute the proportion of points with distances exceeding the threshold
        difference = count_diff / num_known_P
        novel_diff = int(difference > thr_diff)

        return difference, novel_diff, dists

    def ratio_to_neighbors_fC(self, neighbor_dist, thr_diff=0.85):
        count_diff = 0
        #We compute all distances to identify the closest neighbors
        all_dists = []
        for P_i in tqdm(self.list_know_P):
            distance = self.JS.JSDiv_fC(P_i, self.new_Q)
            all_dists.append(distance)
        closests = heapq.nsmallest(self.N, all_dists)
        #We check the proportion of neighbors that are closer that it should be on average
        for dist in closests: 
            if dist >= neighbor_dist:
                count_diff += 1

        #Proportion of neighbor points where the distance is superior to the average distance to normal neighbors -- the higher the more different
        difference = count_diff / len(closests)
        novel_diff = 0
        if difference > thr_diff:
            novel_diff = 1
        mean100 = np.mean(closests)
        return difference, novel_diff, mean100


    def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
        # """
        # Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
        # """
        """
        Compute the proportion of nearest neighbors whose distance to `new_Q` exceeds a given threshold.

        This method measures how many of the closest `N` known distributions (neighbors) differ significantly 
        from the new distribution `new_Q`, using Jensen-Shannon divergence. It provides a measure of local 
        novelty compared to the most similar known examples.

        Parameters:
            neighbor_dist : float
                Distance threshold to compare against for each neighbor.
            thr_diff : float, default=0.85
                Proportion threshold. If more than this proportion of the closest neighbors exceed 
                `neighbor_dist`, the new point is flagged as different.

        Returns:
            difference : float
                Proportion of the N closest neighbors whose distances exceed `neighbor_dist`.
            novel_diff : int
                1 if `difference` > `thr_diff`, else 0.
            mean100 : float
                Mean distance of the N closest neighbors to `new_Q`.
        """
        count_diff = 0
        num_known_P = self.list_know_P.shape[0]
        all_dists = []
        list_know_P = self.list_know_P.tocsr()  # Ensure CSR format for efficiency
        new_Q = self.new_Q

        # Compute distances to all points
        for i in tqdm(range(num_known_P)):
            P_i = list_know_P[i].toarray().flatten()
            all_dists.append(Jensen_Shannon().JSDiv(P_i, new_Q))

        # Identify the closest N neighbors
        closest_dists = heapq.nsmallest(self.N, all_dists)

        # Count neighbors with distances exceeding the threshold
        count_diff = sum(1 for dist in closest_dists if dist > neighbor_dist)

        # Compute the proportion of neighbors exceeding the threshold
        difference = count_diff / len(closest_dists)
        novel_diff = int(difference > thr_diff)
        mean100 = np.mean(closest_dists)

        return difference, novel_diff, mean100
  


    def ratio_to_neighbors_joblib(self, neighbor_dist, thr_diff=0.85):
        # """
        # Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
        # """
        """
        Compute the proportion of nearest neighbors whose distance to `new_Q` exceeds a given threshold, using parallel processing.

        This method uses Joblib to compute Jensen-Shannon distances between `new_Q` and all known distributions in parallel. 
        It then selects the `N` closest neighbors and checks how many of them exceed the `neighbor_dist` threshold.

        Parameters:
            neighbor_dist : float
                Distance threshold to compare each neighbor's distance against.
            thr_diff : float, default=0.85
                Proportion threshold above which the new distribution is flagged as different.

        Returns:
            difference : float
                Proportion of the N nearest neighbors whose distance to `new_Q` exceeds `neighbor_dist`.
            novel_diff : int
                1 if `difference` > `thr_diff`, else 0.
            mean100 : float
                Mean distance among the N nearest neighbors.
        """

        new_Q = self.new_Q
        list_know_P = self.list_know_P
        def compute_distance(i):
            P_i = list_know_P[i].toarray().flatten()
            return Jensen_Shannon().JSDiv(P_i, new_Q)
        
        # Compute distances to all points in parallel
        all_dists = Parallel(n_jobs=-1, batch_size=int(list_know_P.shape[0]/os.cpu_count()))(delayed(compute_distance)(i) for i in (range(list_know_P.shape[0])))
        
        # Identify the closest N neighbors
        closests = heapq.nsmallest(self.N, all_dists)
        count_diff = sum(1 for dist in closests if dist > neighbor_dist)
        
        # Compute the proportion of neighbors exceeding the threshold
        difference = count_diff / len(closests)
        novel_diff = int(difference > thr_diff)
        mean100 = np.mean(closests)

        return difference, novel_diff, mean100 
    

class ClusterKS(Difference):
    """
    Cluster-based extension of the Difference class for detecting distributional shifts using KMeans clustering.

    This class applies KMeans clustering to group known distributions and efficiently estimate whether a new distribution (`new_Q`) 
    differs significantly from those in similar clusters.

    Parameters:
        list_know_P : scipy.sparse matrix
            Matrix of known document distributions (one per row).
        new_Q : numpy.ndarray
            Distribution vector representing a new document or variation.
        N : int
            Number of neighbors to consider when computing neighbor-based distances.
        nbPtsPerCluster : int
            Approximate number of points to assign per KMeans cluster.
    """
    def __init__(self, list_know_P, new_Q, N, nbPtsPerCluster):
        super().__init__(list_know_P, new_Q, N)

        self.nbPtsPerCluster=nbPtsPerCluster

    def clusterKS(self):

        """
        Apply KMeans clustering to the known distributions (`list_know_P`).

        The number of clusters is automatically chosen based on the total number of points and the specified `nbPtsPerCluster`.

        Returns:
            clusters : dict
                Dictionary mapping cluster indices to lists of data point indices.
            kmeans : sklearn.cluster.KMeans
                Trained KMeans object containing the cluster centers.
        """

        # Reduce dimensionality for faster clustering (optional)

        # Perform clustering
        n_clusters = math.ceil(self.list_know_P.shape[0]/self.nbPtsPerCluster)  # Set based on dataset size and structure
        kmeans = KMeans(n_clusters=n_clusters) #, random_state=42)
        # print(n_clusters)
        labels = kmeans.fit_predict(self.list_know_P)

        # Cluster assignments
        clusters = {i: [] for i in (range(n_clusters))}
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        self.clusters = clusters
        # self.pca = pca
        self.kmeans = kmeans
        return clusters, kmeans
    

    def ratio_to_neighbors_kmeans(self, variation_dist, neighbor_dist=0, thr_diff=0.85, nb_clusters=4, spacyUpdated = False):
        """
        Compute the proportion of closest neighbors in selected clusters that are farther than a given distance threshold.

        This method identifies the nearest `nb_clusters` clusters to `variation_dist`, selects all points from those clusters,
        and computes Jensen-Shannon distances to find the closest N neighbors. It then evaluates how many of them exceed 
        `neighbor_dist`.

        Parameters:
            variation_dist : numpy.ndarray
                Target distribution to compare with cluster-based neighbors.
            neighbor_dist : float, default=0
                Distance threshold above which neighbors are considered different.
            thr_diff : float, default=0.85
                Proportion threshold above which the distribution is flagged as different.
            nb_clusters : int, default=4
                Number of closest KMeans clusters to consider.
            spacyUpdate : boolean
                updated sPacy seems to break this code. If true, uses threading in parallelization. Might be slower than before. If false (use pip install pip install spacy==3.7.2), runs like before


        Returns:
            dif_score : float
                Proportion of selected neighbors with distance > `neighbor_dist`.
            dif_bin : int
                1 if `dif_score` > `thr_diff`, else 0.
            mean100 : float
                Mean of the distances among the N closest neighbors in selected clusters.
        """
        # Compute distances from the k-means cluster centers to the target distribution
        cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - variation_dist, axis=1)
        closest_clusters = np.argsort(cluster_dists)[:nb_clusters]

        # Gather indices of points in the closest clusters
        closest_indices = []
        for cluster_idx in closest_clusters:
            closest_indices.extend(self.clusters[cluster_idx])

        # Subset the known probability distributions to those in the closest clusters
        closest_distributions = self.list_know_P[closest_indices]

        def compute_jsd(i):
            """
            Compute the Jensen-Shannon divergence between the target distribution 
            and a specific distribution from the closest clusters.
            """
            P = closest_distributions[i].toarray().flatten()  # Convert sparse row to dense
            return Jensen_Shannon().JSDiv(P=P, Q=variation_dist)

        # Parallelize computation of Jensen-Shannon divergences
        batch_size = max(1, int(closest_distributions.shape[0] / os.cpu_count()))

        ####
        if spacyUpdated == False:
            js_divergences = Parallel(n_jobs=-1, batch_size=batch_size)(
                delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0])
            )
        else:
            with parallel_backend('threading'):
                js_divergences = Parallel(n_jobs=-1)(
                    delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0])
                )
        ####
        
        # Find the smallest N Jensen-Shannon divergences
        kmean_closest = heapq.nsmallest(self.N, js_divergences)
        # print(kmean_closest)
        count_diff = sum(1 for dist in kmean_closest if dist > neighbor_dist)

        dif_score = count_diff / len(kmean_closest)
        dif_bin = int(dif_score > thr_diff)
        mean100 = np.mean(kmean_closest)

        return dif_score, dif_bin, mean100 #, kmean_closest, mean100
    


    
    def dist_estimate_clusters(self, iterations, nb_clusters, spacyUpdated = False):
        """
        Estimate the average distance of a distribution to its closest N neighbors within nearby clusters.

        This function randomly samples distributions from the known set, finds the `nb_clusters` nearest cluster centers, and 
        computes the average of the N smallest Jensen-Shannon distances between the sampled distribution and the selected clusters.

        Parameters:
            iterations : int
                Number of random distributions to sample for distance estimation.
            nb_clusters : int
                Number of closest clusters to consider for each sampled point.
            spacyUpdate : boolean
                updated sPacy seems to break this code. If true, uses threading in parallelization. Might be slower than before. If false (use pip install spacy==3.7.2), runs like before

        Returns:
            mean_distance : float
                Average of mean distances across the `iterations` sampled distributions.
        """
        self.nb_clusters = nb_clusters
        def process_random_point(random_point):
            cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - random_point, axis=1)
            closest_clusters = np.argsort(cluster_dists)[:self.nb_clusters]

            closest_indices = []
            for cluster_idx in closest_clusters:
                closest_indices.extend(self.clusters[cluster_idx])
            closest_distributions = self.list_know_P[closest_indices]

            P = closest_distributions  
            js_divergences = np.array([Jensen_Shannon().JSDiv(P[i].toarray().flatten(), random_point) for i in range(P.shape[0])])
            closest_divergences = heapq.nsmallest(self.N, js_divergences)
            return np.mean(closest_divergences)

        num_points = self.list_know_P.shape[0]
        random_indices = np.random.choice(num_points, size=iterations, replace=False)  # Select random points upfront
        random_points = [self.list_know_P[idx].toarray().flatten() for idx in random_indices]

        # Parallelize across random points
        ####
        if spacyUpdated == False:
            results = Parallel(n_jobs=-1)(
                delayed(process_random_point)(random_point) for random_point in random_points
            )
        else:
            with parallel_backend('threading'):
                results = Parallel(n_jobs=-1)(
                    delayed(process_random_point)(random_point) for random_point in random_points
                )        
        ####

        return np.mean(results)






