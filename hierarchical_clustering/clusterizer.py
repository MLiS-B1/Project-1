"""Implements the make clusters function."""

import heapq
import itertools

import hierarchical_clustering.cluster as cluster
import hierarchical_clustering.cluster_pair as cluster_pair

# Aliases.
Cluster = cluster.Cluster
ClusterPair = cluster_pair.ClusterPair


def make_clusters(points, no_clusters=2, verbose=False):
    """Creates the clusters for the passed in points.

    :param list[tuple] points: The points to clusterize; all of them must
    have the same dimensions.

    :param int no_clusters: The number of clusters to make.

    :param bool verbose: If true it will print debugging messages.

    :return: A list of Cluster instances.
    :rtype: list[Cluster]
    """
    _validate_points(points)
    clusters = [Cluster(*p) for p in points]

    pairs = [
        ClusterPair(c1, c2) for c1, c2 in itertools.combinations(clusters, 2)
    ]

    heapq.heapify(pairs)

    while len(clusters) > no_clusters:
        if verbose:
            print(len(clusters))
        best_pair = heapq.heappop(pairs)

        # Remove all the pairs overlapping with the best pair.
        pairs = [p for p in pairs if not p.overlaps(best_pair)]

        # Remove all the clusters that will be merged.
        clusters = [c for c in clusters if not best_pair.contains_cluster(c)]

        # Create the new cluster.
        new_cluster = best_pair.merge()

        # Add the pairs for the new_cluster.
        for c in clusters:
            pairs.append(ClusterPair(c, new_cluster))

        heapq.heapify(pairs)
        # Add the new cluster to the active clusters.
        clusters.append(new_cluster)
    return clusters


def _validate_points(points):
    """Validates that all points have the same dimensions.

    :raises: AssertionError.
    """
    assert len(points) >= 1
    dimensions = len(points[0])
    for p in points:
        assert dimensions == len(p)




