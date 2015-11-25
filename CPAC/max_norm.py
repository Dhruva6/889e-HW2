import numpy
import scipy

from nearpy.distances.distance import Distance
from scipy.spatial.distance import chebyshev as d

class ChebyshevDistance(Distance):
    """ Euclidean distance """

    def distance(self, x, y):
        """
        Computes distance measure between vectors x and y. Returns float.
        """
        return d(x, y)
