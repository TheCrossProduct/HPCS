from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist


def cartesian_product(arrays):
    r"""
    cartesian_product(arrays)

    Generalized cartesian product of a list of arrays

    Parameters
    ----------
    arrays: list of arrays

    Returns
    -------
    cp: ndarray
        Cartesian product of all arrays in list

    """
    # lenght of list of arrays
    la = len(arrays)
    a = np.atleast_3d(arrays)

    dtype = np.find_common_type([a.dtype for a in arrays], [])

    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)

    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    cp = arr.reshape(-1, la)

    return cp


def set_distance(array1, array2, return_amin=False):
    """
    auxiliary function that compute distance between two sets as
        d(array_1, array_2)  = min_x min_y d(x,y)
    where x and y are repectively elements of array1 and array2

    Parameters
    ----------
    array1: ndarray

    array2: ndarray

    return_amin: bool
        if true it return also the argmin for the distance
    Returns
    -------
    d: float
        distance between array1 and array2
    amin: tuple
        if return_am
    """
    dist_mat = cdist(array1, array2)

    if return_amin:
        m = len(array2)
        min = dist_mat.min()
        amin = dist_mat.argmin()
        i, j = amin // m, amin % m

        return min, (i, j)
    else:
        return dist_mat.min()


def subset_backprojection(bool_map):
    """Util function that given an indicator map of a subset A of X return a map that assign to each index of an elemnt
    in A its index in X.

    Parameters
    ----------
    bool_map: ndarray
        Boolean array. The balue of bool_map[i] = True if i-th element belongs to subset A

    Returns
    -------
    back_bool_map: ndarray
        Map that assign to i-th element of A its idx in the upset X.
    """
    # size of set X
    n = len(bool_map)
    return np.arange(n)[bool_map]


def subset_projection(particel_map, yval):
    """

    Parameters
    ----------
    particel_map: ndarray
        Array representing a mapping from a set X={1...N} to a set Y={1 ... M}, with M <= N

    yval: int
        value in the set {1...M}

    Return
    ------
    proj_map: ndarray
        boolean array whose value is 1 if the element in the X set is such that f(x)=yval
    back_proj_map: ndarray
        array associating each element in the subset X|f(x)=yval to the main set X

    """
    # projection map from the set X to the set {x in X | f(x)=yval }
    proj_map = particel_map == yval

    # back project map
    back_proj_map = subset_backprojection(proj_map)
    return proj_map, back_proj_map
