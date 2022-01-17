"""Implements the ClusterPair class."""

import hierarchical_clustering.cluster as cluster


class ClusterPair:
    """A utility class simplifying the heapify process."""

    def __init__(self, cluster_1, cluster_2):
        """Initializer.

        :param Cluster cluster_1: The first cluster of the pair.
        :param Cluster cluster_2: The second cluster of the pair.
        """
        self._distance = cluster_1.get_distance(cluster_2)
        self._cluster_1 = cluster_1
        self._cluster_2 = cluster_2

    def contains_cluster(self, cluster_to_check):
        """Returns true if the passed in cluster is in the pair.

        :param Cluster cluster_to_check: The cluster to check.

        :return: True if the passed in cluster is in the pair.
        :rtype: bool.
        """
        return (
                cluster_to_check == self._cluster_1 or
                cluster_to_check == self._cluster_2
        )

    def overlaps(self, other):
        """Checks whether other is overlapping with this.

        :param ClusterPair other: The pair to check for overlapping.

        :return: True if other is overlapping with this.
        :rtype: bool.
        """
        if other._cluster_1 == self._cluster_1:
            return True

        if other._cluster_1 == self._cluster_2:
            return True

        if other._cluster_2 == self._cluster_1:
            return True

        if other._cluster_2 == self._cluster_2:
            return True

        return False

    def merge(self):
        """Merges the paired clusters into a new one.

        :return: A new cluster holding all the points for the pair.
        :rtype: Cluster.
        """
        return cluster.Cluster.merge(self._cluster_1, self._cluster_2)

    def __contains__(self, point):
        """Checks is the passed in point is contained in the cluster pair.

        :param tuple point: The point to check if it is contained.

        :return: True if the point is contained.
        :rtype: bool.
        """
        return point in self._cluster_1 or point in self._cluster_2

    def __lt__(self, other):
        """Checks if the distance of other is larger or not.

        :param ClusterPair other: The other pair to compare its distance.

        :return: True if the distance of this pair is smalller than the other's.
        :rtype: bool.
        """
        return self._distance < other._distance

    def __repr__(self):
        """Programmer friendly representation of the instance."""
        return f'ClusterPair with distance: {self._distance}'
