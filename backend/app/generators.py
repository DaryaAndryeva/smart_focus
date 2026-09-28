import math
import networkx as nx
from app.models import GraphType


def generate_graph(
    graph_type: GraphType,
    n: int = 50,
    p: float = 0.1,
    m: int = 2,
    k: int = 4,
    num_edges: int | None = None,
) -> nx.Graph:
    if graph_type == GraphType.DENSE_GNM:
        max_edges = n * (n - 1) // 2
        m_edges = min(num_edges or min(max_edges // 4, n * 10), max_edges)
        if n > 10000:
            return nx.gnm_random_graph(n, m_edges)
        return nx.dense_gnm_random_graph(n, m_edges)

    if graph_type == GraphType.RANDOM:
        if n > 10000:
            p_adj = min(p, 10.0 / n)
        else:
            p_adj = p
        G = nx.erdos_renyi_graph(n, p_adj)
        _ensure_connected(G)
        return G

    if graph_type == GraphType.SCALE_FREE:
        return nx.barabasi_albert_graph(n, min(m, n - 1))

    if graph_type == GraphType.SMALL_WORLD:
        k_actual = min(k, n - 1)
        if k_actual % 2 != 0:
            k_actual -= 1
        k_actual = max(k_actual, 2)
        return nx.watts_strogatz_graph(n, k_actual, p)

    if graph_type == GraphType.POWERLAW_CLUSTER:
        m_actual = min(m, n - 1)
        G = nx.powerlaw_cluster_graph(n, m_actual, p)
        return G

    if graph_type == GraphType.RANDOM_PARTITION:
        num_groups = max(k, 2)
        if n > 10000:
            num_groups = max(num_groups, n // 5000)
        base_size = n // num_groups
        sizes = [base_size] * num_groups
        sizes[-1] += n - base_size * num_groups
        p_in = min(p * 5, 1.0)
        p_out = p * 0.2
        if n > 10000:
            p_in = min(p_in, 50.0 / base_size)
            p_out = min(p_out, 5.0 / base_size)
        G = nx.random_partition_graph(sizes, p_in, p_out)
        return nx.convert_node_labels_to_integers(G)

    if graph_type == GraphType.STOCHASTIC_BLOCK:
        num_blocks = max(k, 2)
        if n > 10000:
            num_blocks = max(num_blocks, n // 5000)
        base_size = n // num_blocks
        sizes = [base_size] * num_blocks
        sizes[-1] += n - base_size * num_blocks
        p_in = min(p * 5, 1.0)
        p_out = p * 0.1
        if n > 10000:
            p_in = min(p_in, 50.0 / base_size)
            p_out = min(p_out, 5.0 / base_size)
        probs = [
            [p_in if i == j else p_out for j in range(num_blocks)]
            for i in range(num_blocks)
        ]
        G = nx.stochastic_block_model(sizes, probs)
        return G

    if graph_type == GraphType.BALANCED_TREE:
        r = max(m, 2)
        h = 1
        while True:
            size = (r ** (h + 1) - 1) // (r - 1)
            if size >= n or h >= 24:
                break
            h += 1
        G = nx.balanced_tree(r, h)
        if G.number_of_nodes() > n:
            G = nx.convert_node_labels_to_integers(G.subgraph(range(n)).copy())
        return G

    if graph_type == GraphType.INTERNET_AS:
        actual_n = min(n, 10000)
        G = nx.random_internet_as_graph(actual_n)
        return G

    if graph_type == GraphType.GRID:
        width = max(math.ceil(math.sqrt(n)), 2)
        height = max(math.ceil(n / width), 2)
        G = nx.grid_2d_graph(height, width)
        if G.number_of_nodes() > n:
            keep = [(r, c) for r in range(height) for c in range(width)][:n]
            G = G.subgraph(keep).copy()
        return nx.convert_node_labels_to_integers(G, ordering="sorted")

    return nx.gnm_random_graph(n, n * 2)


def _ensure_connected(G: nx.Graph) -> None:
    """Connect isolated components by adding minimal edges."""
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return
    main = components[0]
    for comp in components[1:]:
        u = next(iter(main))
        v = next(iter(comp))
        G.add_edge(u, v)
        main = main | comp
