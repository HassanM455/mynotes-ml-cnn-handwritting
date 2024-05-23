import torch
import numpy as np
from typing import (
    Tuple
)

def kmeans(
        data: np.ndarray, 
        num_clusters: int, 
        num_iterations: int = 10
    ) -> Tuple[np.array, np.array]:
    """
    Perform k-means clustering using PyTorch.
    
    :param data: Tensor of shape (num_samples, num_features)
    :param num_clusters: Number of clusters
    :param num_iterations: Number of iterations for k-means
    :return: Tuple of (cluster assignments, centroids)
    """

    data = torch.from_numpy(data).float()

    num_samples, num_features = data.shape

    # Randomly choose clusters from the data points as initial centroids
    indices = torch.randint(num_samples, (num_clusters, ))
    centroids = data[indices]
 

    for _ in range(num_iterations):
        # Compute distances between data points and centroids
        distances = torch.cdist(data, centroids)
        
        # Assign each point to the nearest centroid
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # Update centroids to be the mean of assigned data points
        new_centroids = torch.zeros_like(centroids)
        for i in range(num_clusters):
            points_in_cluster = data[cluster_assignments == i]
            if points_in_cluster.size(0) > 0:
                new_centroids[i] = points_in_cluster.mean(dim=0)
        
        # If centroids don't change, break the loop
        if torch.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids

    
    return np.array(cluster_assignments), np.array(centroids)


