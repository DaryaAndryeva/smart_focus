import logging
import math
import random
import networkx as nx
import numpy as np
from app.models import TopologyType, LayoutAlgorithm
from app.fastlayout import fast_layout

logger = logging.getLogger("smart_focus")

try:
    import igraph as ig
    _HAS_IGRAPH = True
except ImportError:
    _HAS_IGRAPH = False
    from community import community_louvain

SEED = 42


class GraphAnalyzer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._n = graph.number_of_nodes()
        self._m = graph.number_of_edges()
        self._centrality: dict[int, dict] = {}
        self._communities: dict[int, int] | None = None
        self._modularity: float | None = None
        self._global_metrics: dict | None = None
        self._last_positions: dict[int, list[float]] = {}
        self._topology: TopologyType | None = None
        self._clustering: float | None = None
        self._ig = None
        self._ig_nodes: list | None = None

    # ── зеркало графа в igraph ──────────────────────────────────────

    def _igraph(self):
        """Строит (однажды) копию графа в igraph.
        """
        if self._ig is not None:
            return self._ig, self._ig_nodes
        if not _HAS_IGRAPH:
            return None, None
        G = self.graph
        nodes = list(G.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        self._ig = ig.Graph(
            n=len(nodes),
            edges=[(index[u], index[v]) for u, v in G.edges()],
            directed=False,
        )
        self._ig_nodes = nodes
        return self._ig, self._ig_nodes

    # ── global metrics ──────────────────────────────────────────────

    def _avg_clustering(self) -> float:
        """Средний коэффициент кластеризации (кэшируется)."""
        if self._clustering is not None:
            return self._clustering
        igg, _ = self._igraph()
        if igg is not None:
            try:
                self._clustering = float(igg.transitivity_avglocal_undirected(mode="zero"))
                return self._clustering
            except Exception:
                pass
        G = self.graph
        if self._n > 2000:
            self._clustering = nx.average_clustering(
                G, random.Random(SEED).sample(list(G.nodes()), min(500, self._n))
            )
        else:
            self._clustering = nx.average_clustering(G)
        return self._clustering

    def global_metrics(self) -> dict:
        if self._global_metrics is not None:
            return self._global_metrics

        G = self.graph
        n, m = self._n, self._m

        density = nx.density(G)
        avg_degree = (2 * m / n) if n > 0 else 0

        num_components = nx.number_connected_components(G)
        is_connected = nx.is_connected(G)

        clustering = self._avg_clustering()

        diameter = None
        radius = None
        avg_path_length = None

        if is_connected and n <= 500:
            igg, _ = self._igraph()
            try:
                if igg is not None:
                    diameter = int(igg.diameter())
                    radius = int(igg.radius())
                    avg_path_length = float(igg.average_path_length())
                else:
                    diameter = nx.diameter(G)
                    radius = nx.radius(G)
                    avg_path_length = nx.average_shortest_path_length(G)
            except Exception:
                pass

        if self._modularity is None:
            self.detect_communities()
        modularity = self._modularity

        self._global_metrics = {
            "num_nodes": n,
            "num_edges": m,
            "density": round(density, 4),
            "diameter": diameter,
            "radius": radius,
            "clustering_coefficient": round(clustering, 4),
            "num_components": num_components,
            "modularity": round(modularity, 4) if modularity is not None else 0.0,
            "avg_degree": round(avg_degree, 2),
            "avg_path_length": round(avg_path_length, 4) if avg_path_length else None,
        }
        return self._global_metrics

    # ── centrality ──────────────────────────────────────────────────

    def _betweenness(self) -> dict:
        """Центральность посредничества."""
        G = self.graph
        n = self._n
        igg, nodes = self._igraph()

        if igg is not None:
            k = n if n <= 400 else min(max(64, n // 40), 256)
            try:
                if k >= n:
                    raw = igg.betweenness()
                    scale = 1.0
                else:
                    sources = random.Random(SEED).sample(range(n), k)
                    raw = igg.betweenness(sources=sources)
                    scale = n / k
                norm = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
                return {nodes[i]: raw[i] * scale * norm for i in range(n)}
            except Exception:
                pass

        if n > 50000:
            logger.warning(
                "betweenness отключена: n=%d, igraph недоступен — "
                "сохранение мостов при фокусировке работать не будет", n
            )
            return dict.fromkeys(G.nodes(), 0.0)
        if n > 10000:
            return nx.betweenness_centrality(G, k=15)
        if n > 500:
            return nx.betweenness_centrality(G, k=min(max(30, n // 50), 80))
        return nx.betweenness_centrality(G)

    def _closeness(self) -> dict:
        """Центральность близости."""
        igg, nodes = self._igraph()
        if igg is None:
            return nx.closeness_centrality(self.graph)
        n = self._n
        raw = np.nan_to_num(np.array(igg.closeness(normalized=True)))
        comp = igg.connected_components()
        reach = np.array(comp.sizes())[comp.membership]
        vals = raw * (reach - 1) / max(n - 1, 1)
        return {nodes[i]: float(vals[i]) for i in range(n)}

    def _eigenvector(self) -> dict:
        """Собственная центральность."""
        igg, nodes = self._igraph()
        if igg is None:
            return nx.eigenvector_centrality(self.graph, max_iter=200, tol=1e-4)
        vals = np.array(igg.eigenvector_centrality(scale=True))
        norm = np.linalg.norm(vals)
        if norm > 0:
            vals = vals / norm
        return {nodes[i]: float(vals[i]) for i in range(self._n)}

    def calculate_centrality(self) -> dict[int, dict]:
        if self._centrality:
            return self._centrality

        G = self.graph
        n = self._n
        igg, nodes = self._igraph()

        if igg is not None:
            deg = igg.degree()
            degree_c = {nodes[i]: deg[i] / max(n - 1, 1) for i in range(n)}
        else:
            degree_c = nx.degree_centrality(G)
        betweenness_c = self._betweenness()

        if n > 2000:
            closeness_c = dict.fromkeys(G.nodes(), 0.0)
        else:
            try:
                closeness_c = self._closeness()
            except Exception:
                closeness_c = dict.fromkeys(G.nodes(), 0.0)

        if n > 3000:
            eigenvector_c = dict.fromkeys(G.nodes(), 0.0)
        else:
            try:
                eigenvector_c = self._eigenvector()
            except Exception:
                eigenvector_c = dict.fromkeys(G.nodes(), 0.0)

        result = {}
        for node in G.nodes():
            result[node] = {
                "degree_centrality": round(degree_c.get(node, 0), 6),
                "betweenness_centrality": round(betweenness_c.get(node, 0), 6),
                "closeness_centrality": round(closeness_c.get(node, 0), 6),
                "eigenvector_centrality": round(eigenvector_c.get(node, 0), 6),
                "degree": G.degree(node),
            }

        self._centrality = result
        return result

    # ── community detection ─────────────────────────────────────────

    def detect_communities(self) -> dict[int, int]:
        if self._communities is not None:
            return self._communities

        G = self.graph

        try:
            if _HAS_IGRAPH:
                partition, modularity = self._leiden_communities()
            else:
                partition = community_louvain.best_partition(G)
                modularity = community_louvain.modularity(partition, G)
            self._communities = partition
            self._modularity = modularity
        except Exception:
            self._communities = dict.fromkeys(self.graph.nodes(), 0)
            self._modularity = 0.0

        return self._communities

    def _leiden_communities(self) -> tuple[dict[int, int], float]:
        """Обнаружение сообществ алгоритмом Лейдена."""

        igg, nodes = self._igraph()
        if igg is None:
            raise RuntimeError("igraph недоступен")

        ig.set_random_number_generator(random.Random(SEED))
        part = igg.community_leiden(objective_function="modularity", n_iterations=3)
        membership = part.membership
        partition = {nodes[i]: int(c) for i, c in enumerate(membership)}
        return partition, float(igg.modularity(membership))

    # ── topology classification ─────────────────────────────────────

    def classify_topology(self) -> TopologyType:
        if self._topology is not None:
            return self._topology
        self._topology = self._classify()
        return self._topology

    def _is_sparse(self) -> bool:
        # Разреженный граф: средняя степень не больше sqrt(n).
        # Плотность для этого не подходит: при одной и той же средней
        # степени она падает как 1/n, и один порог значит разное на разных n.
        if self._n == 0:
            return True
        avg_degree = 2 * self._m / self._n
        return avg_degree <= math.sqrt(self._n)

    def _classify(self) -> TopologyType:
        G = self.graph
        n, m = self._n, self._m
        is_conn = nx.is_connected(G)
        sparse = self._is_sparse()

        if m == n - 1 and is_conn:
            return TopologyType.TREE

        clustering = self._avg_clustering()

        self.detect_communities()
        modularity = self._modularity or 0

        if modularity > 0.3 and clustering > 0.2:
            return TopologyType.CLUSTERED

        # малый мир: разреженный, высокая кластеризация, короткие пути
        if sparse and clustering > 0.3 and is_conn and n <= 500:
            try:
                avg_path = nx.average_shortest_path_length(G)
                if avg_path < 2 * np.log(max(n, 2)):
                    return TopologyType.SMALL_WORLD
            except Exception:
                pass

        if sparse:
            return TopologyType.SPARSE
        return TopologyType.DENSE

    # ── layout ──────────────────────────────────────────────────────

    def recommend_layout(self, topology: TopologyType) -> LayoutAlgorithm:
        mapping = {
            TopologyType.TREE: LayoutAlgorithm.KAMADA_KAWAI,
            TopologyType.SPARSE: LayoutAlgorithm.KAMADA_KAWAI,
            TopologyType.CLUSTERED: LayoutAlgorithm.COMMUNITY,
            TopologyType.DENSE: LayoutAlgorithm.CIRCULAR,
            TopologyType.SMALL_WORLD: LayoutAlgorithm.SPRING,
        }
        algo = mapping.get(topology, LayoutAlgorithm.SPRING)
        if algo == LayoutAlgorithm.KAMADA_KAWAI and self._n > 500:
            algo = LayoutAlgorithm.MULTILEVEL if self._n > 1000 else LayoutAlgorithm.SPRING
        # круговая укладка только для почти полных графов
        if algo == LayoutAlgorithm.CIRCULAR and nx.density(self.graph) < 0.5:
            algo = LayoutAlgorithm.SPRING
        if algo == LayoutAlgorithm.SPRING and self._n > 2000:
            algo = LayoutAlgorithm.MULTILEVEL
        # Центры кластеров по окружности осмысленны лишь при малом их числе: на
        # ca-AstroPh с 331 сообществом она вырождается в кольцо точек.
        if algo == LayoutAlgorithm.COMMUNITY and self._n > 2000:
            algo = LayoutAlgorithm.MULTILEVEL
        return algo

    def compute_layout(self, algorithm: LayoutAlgorithm) -> dict[int, list[float]]:
        G = self.graph
        n = self._n

        try:
            if algorithm == LayoutAlgorithm.MULTILEVEL:
                pos = fast_layout(G, communities=self.detect_communities())
            elif algorithm == LayoutAlgorithm.SPRING:
                if n > 2000:
                    pos = fast_layout(G, communities=self.detect_communities())
                elif n > 500:
                    pos = nx.spring_layout(G, k=2 / np.sqrt(n), iterations=40)
                else:
                    pos = nx.spring_layout(G, k=2 / np.sqrt(max(n, 1)), iterations=100)
            elif algorithm == LayoutAlgorithm.CIRCULAR:
                pos = self._circular_layout()
            elif algorithm == LayoutAlgorithm.SPECTRAL:
                if n > 3000:
                    pos = fast_layout(G, communities=self.detect_communities())
                else:
                    pos = self._spectral_layout()
            elif algorithm == LayoutAlgorithm.KAMADA_KAWAI:
                if n > 1000:
                    pos = fast_layout(G, communities=self.detect_communities())
                elif n > 500:
                    pos = nx.spring_layout(G, k=2 / np.sqrt(n), iterations=60)
                else:
                    pos = nx.kamada_kawai_layout(G)
            elif algorithm == LayoutAlgorithm.SHELL:
                if n > 10000:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.shell_layout(G)
            elif algorithm == LayoutAlgorithm.COMMUNITY:
                pos = self._community_layout()
            else:
                pos = nx.spring_layout(G, iterations=30)
        except Exception:
            pos = fast_layout(G, communities=self.detect_communities()) if n > 2000 else nx.spring_layout(G, iterations=15)

        result = self._normalize_positions(pos)
        self._last_positions = result
        return result

    def _spectral_layout(self) -> dict:
        """Спектральная укладка с разведением совпавших вершин. На симметричном
        графе (кратные собственные векторы лапласиана) дерево из 1023 вершин
        давало всего 35 различных позиций. Совпавшие вершины расставляются по
        окружности малого радиуса, поэтому структура сохраняется.
        """
        pos = nx.spectral_layout(self.graph)
        if not pos:
            return pos

        groups: dict[tuple[float, float], list] = {}
        for node, p in pos.items():
            key = (round(float(p[0]), 6), round(float(p[1]), 6))
            groups.setdefault(key, []).append(node)

        if len(groups) == len(pos):
            return pos

        coords = np.array([[float(p[0]), float(p[1])] for p in pos.values()])
        span = float(max(coords.max(axis=0) - coords.min(axis=0)))
        if span <= 0:
            span = 1.0

        out = dict(pos)
        for (cx, cy), members in groups.items():
            if len(members) == 1:
                continue
            radius = span * 0.02 * math.sqrt(len(members))
            for i, node in enumerate(sorted(members)):
                angle = 2.0 * math.pi * i / len(members)
                out[node] = [cx + radius * math.cos(angle), cy + radius * math.sin(angle)]
        return out

    def _circular_layout(self) -> dict:
        """Круговая укладка с порядком вершин по сообществам. nx.circular_layout
        расставляет вершины по нумерации, а на дереве это обход по уровням:
        родитель и его дети попадают на противоположные дуги, все рёбра идут
        хордами через центр и рисунок вырождается в клубок.
        """
        communities = self.detect_communities()
        degrees = dict(self.graph.degree())
        nodes = sorted(
            self.graph.nodes(),
            key=lambda nd: (communities.get(nd, 0), -degrees.get(nd, 0), nd),
        )
        count = len(nodes)
        if count == 0:
            return {}
        step = 2.0 * math.pi / count
        return {
            nd: [math.cos(i * step), math.sin(i * step)]
            for i, nd in enumerate(nodes)
        }

    def _community_layout(self) -> dict:
        """Размещает сообщества по окружности, вершины — внутри сообществ.
        Равные углы с общим разбросом min(0.9, 1.8/k) на сети из 332 сообществ
        давали разброс 0.005: кластеры вырождались в точки, а крупные (Лейден
        отдаёт их по убыванию размера) вставали рядом и накладывались.
        """
        communities = self.detect_communities()

        comm_members: dict[int, list[int]] = {}
        for nd, cid in communities.items():
            comm_members.setdefault(cid, []).append(nd)

        comm_ids = sorted(comm_members, key=lambda c: -len(comm_members[c]))
        radii = {cid: np.sqrt(len(comm_members[cid])) for cid in comm_ids}
        total = sum(radii.values()) or 1.0
        ring = max(1.0, 1.15 * total / np.pi)
        max_r = max(radii.values())

        pos: dict = {}
        angle = 0.0
        for cid in comm_ids:
            members = comm_members[cid]
            size = len(members)
            span = 2 * np.pi * radii[cid] / total
            theta = angle + span / 2
            angle += span
            center = np.array([np.cos(theta), np.sin(theta)]) * ring

            # корень из размера: площадь растёт линейно, плотность постоянна
            spread = 0.9 * radii[cid] / max_r * (ring / 2.5)

            sub = self.graph.subgraph(members)
            if size > 400:
                sub_pos = self._scale_unit(fast_layout(sub))
            elif size > 2:
                sub_pos = self._scale_unit(nx.spring_layout(sub, iterations=40))
            else:
                sub_pos = {nd: np.array([0.0, 0.0]) for nd in members}

            for node, p in sub_pos.items():
                pos[node] = center + np.array(p) * spread

        return pos

    def _scale_unit(self, pos: dict) -> dict:
        """Scale arbitrary positions into [-1, 1] x [-1, 1] around the centroid."""
        coords = np.array([[p[0], p[1]] for p in pos.values()], dtype=float)
        coords -= coords.mean(axis=0)
        extent = np.abs(coords).max()
        if extent > 0:
            coords /= extent
        return {node: coords[i] for i, node in enumerate(pos.keys())}

    # ── helpers ──────────────────────────────────────────────────────

    def _normalize_positions(self, pos: dict) -> dict[int, list[float]]:
        if not pos:
            return {}

        coords = np.array([[float(p[0]), float(p[1])] for p in pos.values()])

        # Масштаб по 1-му и 99-му процентилям, а не по крайним точкам: выбросы
        # спектральной укладки сжимали остальной граф в точку (на BA-графе из
        # 300 вершин 56% попадали в центральные 10% площади).
        lo = np.percentile(coords, 1, axis=0)
        hi = np.percentile(coords, 99, axis=0)
        ranges = hi - lo
        degenerate = ranges <= 0
        if degenerate.any():
            span = coords.max(axis=0) - coords.min(axis=0)
            ranges = np.where(degenerate, np.where(span > 0, span, 1.0), ranges)

        pad = 0.05
        normalized = pad + (coords - lo) / ranges * (1 - 2 * pad)
        np.clip(normalized, 0.0, 1.0, out=normalized)

        result = {}
        for i, node in enumerate(pos.keys()):
            result[int(node)] = [round(float(normalized[i, 0]), 6), round(float(normalized[i, 1]), 6)]

        return result

    # ── focus ────────────────────────────────────────────────────────

    def compute_focus(
        self,
        node_id: int,
        depth: int = 2,
        preserve_bridges: bool = True,
        preserve_hubs: bool = True,
        hub_threshold: float = 0.8,
        edge_opacities: bool = True,
    ) -> dict:
        """Вычисляет контекст фокуса. edge_opacities=False пропускает словарь
        прозрачностей рёбер: на графе со 198 050 рёбрами он весит около 4 МБ в
        JSON, а клиент выводит прозрачность ребра из прозрачностей его концов.
        """
        G = self.graph

        if node_id not in G:
            return {
                "node_opacities": {},
                "edge_opacities": {},
                "context_nodes": [],
                "focus_node": node_id,
            }

        distances = nx.single_source_shortest_path_length(G, node_id, cutoff=depth)
        context_nodes = set(distances.keys())

        node_opacities: dict[int, float] = {}
        for node in G.nodes():
            if node in distances:
                dist = distances[node]
                opacity = 1.0 - (dist / (depth + 1)) * 0.3
                node_opacities[node] = round(opacity, 2)
            else:
                node_opacities[node] = 0.06

        if preserve_bridges and self._centrality:
            bw = [c["betweenness_centrality"] for c in self._centrality.values()]
            if bw:
                thresh = float(np.percentile(bw, 90))
                for nd, cent in self._centrality.items():
                    if cent["betweenness_centrality"] >= thresh and nd not in context_nodes:
                        node_opacities[nd] = max(node_opacities.get(nd, 0), 0.3)

        if preserve_hubs and self._centrality:
            dg = [c["degree_centrality"] for c in self._centrality.values()]
            if dg:
                thresh = float(np.percentile(dg, hub_threshold * 100))
                for nd, cent in self._centrality.items():
                    if cent["degree_centrality"] >= thresh and nd not in context_nodes:
                        node_opacities[nd] = max(node_opacities.get(nd, 0), 0.25)

        edge_op: dict[str, float] = {}
        if edge_opacities:
            for u, v in G.edges():
                key = f"{u}-{v}"
                if u in context_nodes and v in context_nodes:
                    mx = max(distances.get(u, depth + 1), distances.get(v, depth + 1))
                    edge_op[key] = round(1.0 - (mx / (depth + 1)) * 0.3, 2)
                else:
                    edge_op[key] = 0.03

        return {
            "node_opacities": node_opacities,
            "edge_opacities": edge_op,
            "context_nodes": sorted(context_nodes),
            "focus_node": node_id,
        }

    def community_sizes(self) -> dict[int, int]:
        communities = self.detect_communities()
        sizes: dict[int, int] = {}
        for cid in communities.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        return sizes

    # ── collapse ─────────────────────────────────────────────────────

    def collapse_communities(self, min_size: int = 10) -> tuple[list[dict], list[dict]]:
        """Collapse communities with >= min_size nodes into single meta-nodes."""
        communities = self.detect_communities()
        centrality = self.calculate_centrality()
        positions = self._last_positions

        comm_groups: dict[int, list[int]] = {}
        for node, cid in communities.items():
            comm_groups.setdefault(cid, []).append(node)

        collapsed_comms = {
            cid for cid, members in comm_groups.items()
            if len(members) >= min_size
        }

        node_mapping: dict[int, int] = {}
        meta_id = -1
        meta_id_for_comm: dict[int, int] = {}

        nodes_out: list[dict] = []

        for cid, members in comm_groups.items():
            if cid in collapsed_comms:
                mid = meta_id
                meta_id_for_comm[cid] = mid
                meta_id -= 1
                for nd in members:
                    node_mapping[nd] = mid

                xs = [positions[n][0] for n in members if n in positions]
                ys = [positions[n][1] for n in members if n in positions]
                cx = sum(xs) / len(xs) if xs else 0.5
                cy = sum(ys) / len(ys) if ys else 0.5

                deg_c = np.mean([centrality[n]["degree_centrality"] for n in members])
                btw_c = np.mean([centrality[n]["betweenness_centrality"] for n in members])
                max_deg = max(centrality[n]["degree"] for n in members)

                nodes_out.append({
                    "id": mid,
                    "label": f"{len(members)} узлов",
                    "x": cx,
                    "y": cy,
                    "degree_centrality": round(float(deg_c), 6),
                    "betweenness_centrality": round(float(btw_c), 6),
                    "closeness_centrality": 0.0,
                    "eigenvector_centrality": 0.0,
                    "community": cid,
                    "degree": max_deg,
                    "is_collapsed": True,
                    "member_count": len(members),
                })
            else:
                for nd in members:
                    node_mapping[nd] = nd
                    cent = centrality.get(nd, {})
                    pos = positions.get(nd, [0.5, 0.5])
                    nodes_out.append({
                        "id": nd,
                        "label": str(nd),
                        "x": pos[0],
                        "y": pos[1],
                        "degree_centrality": cent.get("degree_centrality", 0),
                        "betweenness_centrality": cent.get("betweenness_centrality", 0),
                        "closeness_centrality": cent.get("closeness_centrality", 0),
                        "eigenvector_centrality": cent.get("eigenvector_centrality", 0),
                        "community": cid,
                        "degree": cent.get("degree", 0),
                        "is_collapsed": False,
                        "member_count": 1,
                    })

        edge_weights: dict[tuple[int, int], int] = {}
        for u, v in self.graph.edges():
            mu = node_mapping.get(u, u)
            mv = node_mapping.get(v, v)
            if mu == mv:
                continue
            key = (min(mu, mv), max(mu, mv))
            edge_weights[key] = edge_weights.get(key, 0) + 1

        edges_out = [
            {"source": k[0], "target": k[1], "weight": w}
            for k, w in edge_weights.items()
        ]

        return nodes_out, edges_out
