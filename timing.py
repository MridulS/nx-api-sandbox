import timeit
import grblas as gb
import matplotlib.pyplot as plt
import networkx as nx
import tqdm
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python

from nx_api import pagerank


def timeout(p, total_nodes=1000, jumps=100):
    timer_p = list()
    timer_n = list()
    timer_s = list()
    timer_g = list()
    new_p, new_n, new_s, new_g = list(), list(), list(), list()
    nodes = list(range(jumps, total_nodes, jumps))
    for n in tqdm.tqdm(nodes):
        G = nx.erdos_renyi_graph(n, p)
        nodelist = list(G)
        #         N = nx.to_numpy_array(G)
        S = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, dtype=float)
        M = gb.io.from_networkx(G, nodelist=nodelist, dtype=float)

        #         start = timeit.default_timer()
        #         _pagerank_python(G)
        #         end = timeit.default_timer()
        #         timer_p.append(end - start)

        #         start = timeit.default_timer()
        #         nx.pagerank_numpy(G)
        #         end = timeit.default_timer()
        #         timer_n.append(end - start)

        #         start = timeit.default_timer()
        #         current_scipy_res = nx.pagerank_scipy(G)
        #         end = timeit.default_timer()
        #         timer_s.append(end - start)

        #         start = timeit.default_timer()
        #         pagerank(G)
        #         end = timeit.default_timer()
        #         new_p.append(end - start)

        #         start = timeit.default_timer()
        #         pagerank(N, nodelist)
        #         end = timeit.default_timer()
        #         new_n.append(end - start)

        start = timeit.default_timer()
        new_scipy_res = pagerank(S, nodelist)
        end = timeit.default_timer()
        new_s.append(end - start)

        start = timeit.default_timer()
        grblas_time = pagerank(M)
        end = timeit.default_timer()
        new_g.append(end - start)

    #         assert current_scipy_res == new_scipy_res
    # return timer_p, timer_n, timer_s, new_p, new_n, new_s
    #     plt.plot(list(range(100, total_nodes, 100)), timer_p, label="Dicts (current)")
    #     plt.plot(
    #         list(range(100, total_nodes, 100)), timer_n, label="NumPy Arrays (current)"
    #     )
    plt.plot(
        #         list(range(100, total_nodes, jumps)), timer_s, label="SciPy Sparse (current)"
    )
    #     plt.plot(list(range(100, total_nodes, 100)), new_p, label="Dicts (API)")
    #     plt.plot(list(range(100, total_nodes, jumps)), new_n, label="NumPy Arrays (API)")
    plt.plot(list(range(jumps, total_nodes, jumps)), new_s, label="SciPy Sparse (API)")
    plt.plot(list(range(jumps, total_nodes, jumps)), new_g, label="Grblas (API)")

    plt.legend()
    plt.show()
