from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import random as random
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from itertools import compress
import gzip



def read_data(FOLD):
    
    with gzip.open(f'data/train{FOLD}.gz', "rb") as fd:
        content = [line.decode('utf-8').strip().split(' ') for line in fd]

    num_samples = len(content)
    num_features = len(content[0]) - 1

    labels = np.empty(num_samples, dtype=int)
    features = np.empty((num_samples, num_features), dtype=float)

    for i, row in enumerate(content):
        labels[i] = int(row[0])
        for j, feat in enumerate(row[1:]):
            features[i, j] = float(feat.split(':')[1])

    X = features
    y = labels

    return(X,y)

def ne_dist(labels, distances):
    '''
    Nearest Enemy computation.
    '''
    labels = np.array(labels)
    n = len(labels)
    ne = np.full(n, -1, dtype=int)

    # Create a mask where entries are True if the pair has different labels
    enemy_mask = labels[:, None] != labels[None, :]

    # Set same-class distances to np.inf to ignore them
    masked_distances = np.where(enemy_mask, distances, np.inf)

    # Find the index of the minimum distance in each row
    ne = np.argmin(masked_distances, axis=1)

    return ne

def compute_local_sets(X, y, kept_indices=None):
    '''
    Compute local sets and sort them by increasing Local Set Cardinality (LSC),
    restricted to instances in `kept_indices`.

    Parameters:
    - labels: (n,) full label array
    - ne: (n,) nearest enemy indices in original space
    - distances: (n, n) full pairwise distance matrix
    - kept_indices: list or array of indices to keep (subset of full data)

    Returns:
    - LSs: dict of local sets for each kept index
    - sorted_LSs_dict: dict of kept index → local set (sorted by LSC)
    '''

    if(kept_indices is None):
        kept_indices = np.arange(len(y), dtype=int) # take all instances
    else:
        kept_indices = np.array(kept_indices)

    X_filtered = X[kept_indices]
    y_filtered = y[kept_indices]

    distances_filtered = euclidean_distances(X_filtered, X_filtered)
    NEs_filtered = ne_dist(y_filtered, distances_filtered)
    
    LSs = defaultdict(list)
    lsc_map = []

    for i, real_index in enumerate(kept_indices):
        same_class = (y_filtered == y_filtered[i])
        closer_than_enemy = (distances_filtered[i] <= distances_filtered[i][NEs_filtered[i]])
        mask = same_class & closer_than_enemy
        LS = kept_indices[mask].tolist()

        # Ensure i is included if appropriate
        if real_index not in LS:
            LS.append(real_index)

        LSs[int(real_index)] = LS
        lsc_map.append((int(real_index), len(LS)))

    # Sort by increasing local set size
    lsc_map.sort(key=lambda x: x[1])

    # Create sorted dict
    sorted_LSs_dict = defaultdict(list)
    for i, _ in lsc_map:
        sorted_LSs_dict[int(i)] = LSs[int(i)]

    return LSs, sorted_LSs_dict

def local_set_based_smoother(X, y):
    """
    Local Set Based Smoother (noise filtering).
    
    Keeps points where the number of local sets the point appears in 
    is greater than the number of times it is a nearest enemy.

    Parameters:
    - X: (n_samples, n_features) data array
    - y: (n_samples,) label array
    - ne: (n_samples,) index of nearest enemy for each point
    - distances: (n_samples, n_samples) full pairwise distance matrix

    Returns:
    - kept_indices: indices of retained points (relative to original X)
    """

    n = len(y)
    LSs, _ = compute_local_sets(X, y)
    distances = euclidean_distances(X, X)
    NEs = ne_dist(y, distances)

    # Count how many local sets each point appears in
    usefulness = np.zeros(n, dtype=int)
    for LS in LSs.values():
        for idx in LS:
            usefulness[idx] += 1

    # Count how many times each point is the nearest enemy
    harmfulness = np.zeros(n, dtype=int)
    for enemy in NEs:
        harmfulness[enemy] += 1

    # Apply the smoother condition
    mask = usefulness >= harmfulness
    kept_indices = np.where(mask)[0]
    noise_indices= np.where(np.logical_not(mask))[0]

    return kept_indices, noise_indices

def main():
    FOLD = 0
    X, y = read_data(FOLD)
    LSs, sorted_LSs_dict = compute_local_sets(X, y)
    kept_indices, noise_indices = local_set_based_smoother(X, y)


if __name__ == '__main__':
    main()

