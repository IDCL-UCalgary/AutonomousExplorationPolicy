import numpy as np
from typing import Tuple, List, Optional
from sklearn.cluster import KMeans
from scipy.ndimage import label
from environments.config import map_height, map_width, rmax, number_of_clusters


def frontier_detection(map_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    x_frontier = []
    y_frontier = []

    padded_map = np.pad(map_matrix, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    mask = (map_matrix < 0.5) & (
        (padded_map[:-2, 1:-1] == 0.5) |  # Top neighbor
        (padded_map[1:-1, 2:] == 0.5)  |  # Right neighbor
        (padded_map[2:, 1:-1] == 0.5)  |  # Bottom neighbor
        (padded_map[1:-1, :-2] == 0.5)    # Left neighbor
    )

    y_frontier, x_frontier = np.where(mask)

    final_x = []
    final_y = []
    for i in range(len(x_frontier)):
        if (x_frontier[i] > 0 and x_frontier[i] < map_width-1 and 
            y_frontier[i] > 0 and y_frontier[i] < map_height-1):
            final_x.append(x_frontier[i])
            final_y.append(y_frontier[i])

    final_x, final_y = filter_frontier_points(final_x, final_y, map_matrix)

    return final_x, final_y


def filter_frontier_points(x_frontier, y_frontier, map_matrix):
    labeled, _ = label(map_matrix == 0)
    component_sizes = np.bincount(labeled.flat)
    component_sizes[0] = 0  # Ignore the background label in component sizes
    largest_component = np.argmax(component_sizes)
    filtered_x_frontier = []
    filtered_y_frontier = []
    for x, y in zip(x_frontier, y_frontier):
        if labeled[y, x] == largest_component:
            filtered_x_frontier.append(x)
            filtered_y_frontier.append(y)

    return filtered_x_frontier, filtered_y_frontier


def clustering_filtering(xloc: List[int], yloc: List[int], m: np.ndarray, 
                         num_cluster: int, x: np.ndarray) -> Tuple[List[int], List[int]]:
    
    if not xloc or not yloc:  # Check if frontier lists are empty
        return [], []
        

    X = np.array([xloc, yloc]).T
    actual_clusters = min(num_cluster, len(X))
    if actual_clusters == 0:
        return [], []
    
    model = KMeans(n_clusters=actual_clusters)
    model.fit(X)
    yhat = model.predict(X)
    clusters = np.unique(yhat)
    
    x_frontier = []
    y_frontier = []
    
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)[0]
        mean_x = np.mean(X[row_ix, 0])
        mean_y = np.mean(X[row_ix, 1])
        centroid = [mean_y, mean_x]
        
        mean_y, mean_x = find_nearest_free_space([yloc, xloc], centroid)
        x_frontier.append(int(mean_x))
        y_frontier.append(int(mean_y))
    
    if len(x_frontier) < num_cluster:
        x_frontier, y_frontier = repeat_cluster(x_frontier, y_frontier, num_cluster)
    
    return x_frontier, y_frontier


def find_nearest_free_space(frontiers: List, centroid: List) -> Tuple[float, float]:
    
    frontiers_array = np.column_stack((frontiers[0], frontiers[1]))
    
    distances = np.linalg.norm(
        np.array(frontiers_array) - np.array([centroid[0], centroid[1]]), 
        axis=1
    )
    
    closest_idx = np.argmin(distances)
    return frontiers_array[closest_idx][0], frontiers_array[closest_idx][1]


# def repeat_cluster(actual_clusters_x: List[int], actual_clusters_y: List[int], 
#                   number_of_desired_clusters: int) -> Tuple[List[int], List[int]]:
    
#     n = number_of_desired_clusters - len(actual_clusters_x)
#     x_frontiers = np.tile(actual_clusters_x, n + 1)
#     x_frontiers = x_frontiers[0:number_of_desired_clusters]
    
#     y_frontiers = np.tile(actual_clusters_y, n + 1)
#     y_frontiers = y_frontiers[0:number_of_desired_clusters]
    
#     return x_frontiers, y_frontiers


def frontier_space(m: np.ndarray, number_of_cluster: int, x: np.ndarray) -> Tuple[List[int], List[int]]:
    xloc, yloc = frontier_detection(m)
    if len(xloc) < number_of_cluster:
        x_frontier = xloc
        y_frontier = yloc
    else:
        x_frontier, y_frontier = clustering_filtering(xloc, yloc, m, number_of_cluster, x)
    
    return x_frontier, y_frontier


# def frontier_space_nonlearning(m: np.ndarray, number_of_cluster: int, x: np.ndarray) -> Tuple[List[int], List[int]]:
    
#     x_frontier = []
#     y_frontier = []
    
#     padded_map = np.pad(m, ((1, 1), (1, 1)), mode='constant', constant_values=0)

#     mask = (m < 0.5) & (
#         (padded_map[:-2, 1:-1] == 0.5) |  # Top
#         (padded_map[1:-1, 2:] == 0.5)  |  # Right
#         (padded_map[2:, 1:-1] == 0.5)  |  # Bottom
#         (padded_map[1:-1, :-2] == 0.5)     # Left
#     )

#     y_frontier, x_frontier = np.where(mask)

#     if len(x_frontier) <= number_of_cluster:
#         return x_frontier.tolist(), y_frontier.tolist()

#     # Filter out boundary points
#     final_x = []
#     final_y = []
#     for i in range(len(x_frontier)):
#         if (x_frontier[i] > 0 and x_frontier[i] < map_width-1 and 
#             y_frontier[i] > 0 and y_frontier[i] < map_height-1):
#             final_x.append(x_frontier[i])
#             final_y.append(y_frontier[i])
    
#     # Prepare for clustering
#     X = np.array([final_x, final_y]).T
    
#     # Apply K-means clustering
#     model = KMeans(n_clusters=number_of_cluster)
#     model.fit(X)
#     yhat = model.predict(X)
#     clusters = np.unique(yhat)

#     # Initialize output lists
#     x_frontier = []
#     y_frontier = []
    
#     # For each cluster, find centroid
#     for cluster in clusters:
#         row_ix = np.where(yhat == cluster)[0]
#         mean_x = np.mean(X[row_ix, 0])
#         mean_y = np.mean(X[row_ix, 1])
#         centroid = [mean_y, mean_x]
        
#         # Find nearest frontier to centroid
#         mean_y, mean_x = find_nearest_free_space([final_y, final_x], centroid)
#         x_frontier.append(int(mean_x))
#         y_frontier.append(int(mean_y))
    
#     return x_frontier, y_frontier