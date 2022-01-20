"""Implements the cluster class."""

import copy
import math


class Cluster:
    """A cluster to be used for the hierarchical clusterization.

    :ivar int _dimensions: Holds the dimensions of each point in the cluster.
    :ivar set _points: A set with the points of the cluster.
    :ivar tuple [float] _centroid: Memoize the centroid of the cluster.
    """

    _dimensions = None
    _centroid = None
    _points = None

    def __init__(self, *coordinates):
        """Initializer.

        Note that a cluster cannot be empty, meaning it must contain at least
        one point; this is why if no coordinates will be passed an exception
        will be raised. For the whole lifespan of the instance any point that
        will be added must have the dimensions as the initial.

        :param coordinates: A tuple holding the coordinates of the point to add.
        """
        assert coordinates, "You must pass valid coordinates."
        self._points = set()
        self.add_point(*coordinates)

    def get_features(self, index):
        """Returns the features for the passed in index.

        Example:

        Assuming that your points look like this:
                    [1, 3, 4]
                    [2, 7, 11]
        meaning that they represent three-dimensional data points (x, y, z)
        then passing an index of 0 we will get back the vector <1, 2> while
        for 1 <3, 7> etc.

        :param int index: The index of the features to return.

        :return: A list of doubles representing the features for a specific
        dimension.

        :rtype: list[float]
        """
        return [p[index] for p in self._points]


    def __eq__(self, other):
        """Checks if the passed in cluster has the same points.

        :param Cluster other: The other instance to check.
        :rtype: bool.
        """
        return len(self._points.difference(other._points)) == 0 and len(
            self._points) == len(other._points)

    def __contains__(self, point):
        """Checks is the passed in point is contained in the cluster.

        :param tuple point: The point to check if it is contained.

        :return: True if the point is contained.
        :rtype: bool.
        """
        return point in self._points

    def __len__(self):
        """Returns the number of points in the cluster.

        :return: The number of points in the cluster.
        :rtype: int.
        """
        return len(self._points)

    def add_point(self, *coordinates):
        """Adds a point to the cluster.

        A point is represented as a tuple of floats; the very first time
        this method is called the number of the coordinates will characterize
        the dimensions of each point for the whole lifespan of the instance.

        :param coordinates: A tuple holding the coordinates of the point to add.
        """
        self._centroid = None  # Will force a recalculation of centroid.
        if self._dimensions is None:
            assert len(self._points) == 0
            self._dimensions = len(coordinates)
        assert len(coordinates) == self._dimensions
        assert coordinates not in self._points
        self._points.add(coordinates)

    def get_distance(self, other):
        """Returns the EuclideanDistance to the passed in Cluster.

        Uses the average distance for all the points. Note that there are
        other ways to specify the distance but since this class is meant
        for educational purposes it only supports one distance approach.

        :param Cluster other: The cluster to return the distance.

        :return: The EuclideanDistance between this instance and the passed
        in other instance.

        :rtype: float.
        """
        point_1 = self.get_centroid()
        point_2 = other.get_centroid()
        return self.get_euclidean_distance(point_1, point_2)

    def get_centroid(self):
        """Calculates and returns the centroid of the cluster.

        For a small performance gain the centroid is memoized until
        a new point will be added.

        :return: The centroid point as a tuple of floats.
        :rtype: tuple[float].
        """
        if self._centroid is None:
            assert len(self._points) > 0
            totals = [0] * self._dimensions
            for point in self._points:
                for dim in range(self._dimensions):
                    totals[dim] += point[dim]
            self._centroid = tuple([t / len(self._points) for t in totals])
        return self._centroid

    @classmethod
    def get_euclidean_distance(cls, p1, p2):
        """Calculates the euclidean distance for the passed in points.

        :return: The EuclideanDistance between the passed in points.
        :rtype: float.
        """
        assert len(p1) == len(p2)
        result = 0.0
        for i in range(len(p1)):
            result += pow(float(p1[i]) - float(p2[i]), 2)
        return math.sqrt(result)

    @classmethod
    def merge(cls, cluster_1, cluster_2):
        """Merges the passed in clusters to a new cluster.

        :param Cluster cluster_1: The first cluster to merge.
        :param Cluster cluster_2: The second cluster to merge.

        :return: A new cluster containing all the points for both clusters.
        :rtype: Cluster.
        """
        assert cluster_1._dimensions == cluster_2._dimensions

        new_cluster = copy.deepcopy(cluster_1)
        new_cluster._points.update(cluster_2._points)
        return new_cluster
