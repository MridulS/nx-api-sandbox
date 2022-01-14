from functools import singledispatch

import networkx
import networkx as nx
import numpy
import numpy as np
import scipy as sp
import scipy.sparse
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python


# Dispatch to normal pagerank
@singledispatch
def pagerank(G: networkx.Graph):
    return _pagerank_python(G)


# Dispatch to numpy pagerank
@pagerank.register
def _(G: numpy.ndarray, nodelist):
    M = google_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = float(largest.sum())
    return dict(zip(nodelist, map(float, largest / norm)))


# Dispatch to scipy sparse pagerank
@pagerank.register
def _(
    G: scipy.sparse.csr.csr_matrix,
    nodelist,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    N = len(nodelist)
    M = G
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x = x / x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p = p / p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


@singledispatch
def google_matrix(
    G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None
):
    import numpy as np

    if nodelist is None:
        nodelist = list(G)

    A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
    N = len(G)
    if N == 0:
        # TODO: Remove np.asmatrix wrapper in version 3.0
        return np.asmatrix(A)

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(A.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    A[dangling_nodes] = dangling_weights

    A /= A.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

    # TODO: Remove np.asmatrix wrapper in version 3.0
    return np.asmatrix(alpha * A + (1 - alpha) * p)


@google_matrix.register
def _(
    G: numpy.ndarray,
    alpha=0.85,
    personalization=None,
    nodelist=None,
    weight="weight",
    dangling=None,
):
    A = G
    N = len(G)
    if N == 0:
        # TODO: Remove np.asmatrix wrapper in version 3.0
        return np.asmatrix(A)

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(A.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    A[dangling_nodes] = dangling_weights

    A /= A.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

    # TODO: Remove np.asmatrix wrapper in version 3.0
    return np.asmatrix(alpha * A + (1 - alpha) * p)
