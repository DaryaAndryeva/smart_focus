import time
import logging
import itertools

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
import networkx as nx

from app.models import (
    GenerateRequest, FocusRequest, LayoutRequest,
    CollapseRequest, SubgraphRequest,
)
from app.generators import generate_graph
from app.analytics import GraphAnalyzer
from app.binary import encode_graph, encode_arrays, encode_focus

logger = logging.getLogger("smart_focus")

app = FastAPI(
    title="Smart Focus API",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    description=(
        "API системы адаптивной визуализации графов Smart Focus.\n\n"
        "Система автоматически анализирует топологию графа, классифицирует его структуру, "
        "выбирает оптимальный алгоритм укладки и реализует управление фокусом — "
        "подсветку контекста выбранного узла с сохранением структурно важных элементов (мостов и хабов).\n\n"
        "**Поддерживаемые размеры:** до 100 000 вершин."
    ),
    openapi_tags=[
        {
            "name": "graph",
            "description": "Генерация, загрузка графа и смена алгоритма укладки.",
        },
        {
            "name": "focus",
            "description": "Управление фокусом: выделение контекста выбранного узла, "
                           "затенение нерелевантных элементов, сохранение мостов и хабов.",
        },
        {
            "name": "system",
            "description": "Служебные эндпоинты.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph: nx.Graph | None = None
_analyzer: GraphAnalyzer | None = None
_sub_analyzer: GraphAnalyzer | None = None


def _get_analyzer() -> GraphAnalyzer:
    if _analyzer is None:
        raise HTTPException(status_code=400, detail="Граф не загружен. Сначала сгенерируйте или загрузите граф.")
    return _analyzer


def _active_analyzer() -> GraphAnalyzer:
    """Return subgraph analyzer if viewing a community, otherwise the main one."""
    return _sub_analyzer if _sub_analyzer is not None else _get_analyzer()


# ── бинарный протокол ────────────────────────────────────────────────
# JSON-граф ca-AstroPh (18 772 вершины, 198 050 рёбер) — 13.0 МБ и 0.48 с на 216 822 объекта Pydantic.
# Бинарный формат — 3.1 МБ и 0.3 мс, читается сразу в типизированные массивы для WebGL и воркера.

BIN_MEDIA = "application/octet-stream"


def _binary_payload(analyzer: GraphAnalyzer, layout_algo=None) -> bytes:
    t0 = time.time()
    topology = analyzer.classify_topology()
    recommended = analyzer.recommend_layout(topology)
    algo = layout_algo or recommended
    t1 = time.time()
    positions = analyzer.compute_layout(algo)
    t2 = time.time()

    G = analyzer.graph
    centrality = analyzer.calculate_centrality()
    communities = analyzer.detect_communities()
    nodes = list(G.nodes())
    n = len(nodes)

    ids = np.fromiter(nodes, dtype=np.int64, count=n)
    cols: dict[str, np.ndarray] = {
        "id": ids,
        "x": np.fromiter((positions.get(v, (0.5, 0.5))[0] for v in nodes), np.float32, n),
        "y": np.fromiter((positions.get(v, (0.5, 0.5))[1] for v in nodes), np.float32, n),
        "degree": np.fromiter((centrality[v]["degree"] for v in nodes), np.uint32, n),
        "community": np.fromiter((communities.get(v, 0) for v in nodes), np.uint32, n),
        "member_count": np.ones(n, dtype=np.uint32),
        "is_collapsed": np.zeros(n, dtype=np.uint8),
    }
    for key in ("degree_centrality", "betweenness_centrality",
                "closeness_centrality", "eigenvector_centrality"):
        cols[key] = np.fromiter((centrality[v][key] for v in nodes), np.float32, n)

    # таблица прямого доступа вместо словаря: на порядок быстрее на 198 тысяч обращений
    m = G.number_of_edges()
    flat = np.fromiter(itertools.chain.from_iterable(G.edges()), dtype=np.int64, count=2 * m)
    lookup = np.zeros(int(ids.max()) + 2, dtype=np.uint32)
    lookup[ids] = np.arange(n, dtype=np.uint32)
    pairs = lookup[flat].reshape(-1, 2)
    cols["source"] = pairs[:, 0]
    cols["target"] = pairs[:, 1]
    cols["weight"] = np.ones(m, dtype=np.float32)

    metrics = analyzer.global_metrics()
    t3 = time.time()

    blob = encode_arrays(
        cols, n=n, m=m,
        global_metrics=metrics,
        topology=topology.value,
        recommended_layout=recommended.value,
        community_sizes=analyzer.community_sizes(),
    )
    logger.info(
        "[bin %d nodes] topo=%.2fs layout=%.2fs cols=%.2fs encode=%.2fs -> %.2f MB",
        n, t1 - t0, t2 - t1, t3 - t2, time.time() - t3, len(blob) / 1e6,
    )
    return blob


def _bin(blob: bytes) -> Response:
    return Response(content=blob, media_type=BIN_MEDIA)


@app.post("/api/graph/generate.bin", tags=["graph"], summary="Сгенерировать граф (бинарный ответ)")
def api_generate_bin(req: GenerateRequest):
    global _graph, _analyzer, _sub_analyzer
    _graph = generate_graph(req.graph_type, req.n, req.p, req.m, req.k, req.num_edges)
    _analyzer = GraphAnalyzer(_graph)
    _sub_analyzer = None
    return _bin(_binary_payload(_analyzer))


@app.post(
    "/api/graph/upload.bin",
    tags=["graph"],
    summary="Загрузить граф списком рёбер (бинарный обмен)",
    description=(
        "Тело запроса — сырой массив Uint32 пар вершин (little-endian), "
        "ответ — бинарное представление графа. Избавляет от JSON-массива "
        "на 400 тысяч элементов в обе стороны."
    ),
)
async def api_upload_bin(request: Request):
    global _graph, _analyzer, _sub_analyzer
    raw = await request.body()
    if len(raw) < 8 or len(raw) % 8 != 0:
        raise HTTPException(status_code=400, detail="Ожидался массив пар uint32.")
    pairs = np.frombuffer(raw, dtype="<u4").reshape(-1, 2)
    G = nx.Graph()
    G.add_edges_from(map(tuple, pairs.tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="Пустой граф — ни одного ребра не распознано.")
    _graph = G
    _analyzer = GraphAnalyzer(_graph)
    _sub_analyzer = None
    return _bin(_binary_payload(_analyzer))


@app.post("/api/graph/layout.bin", tags=["graph"], summary="Сменить укладку (бинарный ответ)")
def api_layout_bin(req: LayoutRequest):
    global _sub_analyzer
    _sub_analyzer = None
    return _bin(_binary_payload(_get_analyzer(), req.algorithm))


@app.post("/api/graph/collapse.bin", tags=["graph"], summary="Свернуть сообщества (бинарный ответ)")
def api_collapse_bin(req: CollapseRequest):
    global _sub_analyzer
    _sub_analyzer = None
    analyzer = _get_analyzer()
    if not analyzer._last_positions:
        raise HTTPException(status_code=400, detail="Сначала выполните генерацию или загрузку графа.")
    nodes, edges = analyzer.collapse_communities(req.min_size)
    topology = analyzer.classify_topology()
    return _bin(encode_graph(
        nodes, edges,
        global_metrics=analyzer.global_metrics(),
        topology=topology.value,
        recommended_layout=analyzer.recommend_layout(topology).value,
        community_sizes=analyzer.community_sizes(),
    ))


@app.post("/api/graph/subgraph.bin", tags=["graph"], summary="Открыть сообщество (бинарный ответ)")
def api_subgraph_bin(req: SubgraphRequest):
    global _sub_analyzer
    analyzer = _get_analyzer()
    communities = analyzer.detect_communities()
    members = [n for n, cid in communities.items() if cid == req.community_id]
    if not members:
        raise HTTPException(status_code=404, detail=f"Сообщество {req.community_id} не найдено.")
    _sub_analyzer = GraphAnalyzer(analyzer.graph.subgraph(members).copy())
    return _bin(_binary_payload(_sub_analyzer))


@app.post(
    "/api/graph/focus.bin",
    tags=["focus"],
    summary="Контекст фокуса (бинарный ответ)",
    description=(
        "Возвращает только прозрачности вершин массивом float32. "
        "Прозрачность рёбер клиент выводит из концевых вершин прямо в "
        "шейдере — это экономит около 4 МБ на графе со 198 050 рёбрами."
    ),
)
def api_focus_bin(req: FocusRequest):
    analyzer = _active_analyzer()
    result = analyzer.compute_focus(
        node_id=req.node_id,
        depth=req.depth,
        preserve_bridges=req.preserve_bridges,
        preserve_hubs=req.preserve_hubs,
        hub_threshold=req.hub_threshold,
        edge_opacities=False,
    )
    return _bin(encode_focus(
        [int(n) for n in analyzer.graph.nodes()],
        result["node_opacities"],
        result["context_nodes"],
        result["focus_node"],
    ))


@app.post(
    "/api/graph/focus/reset",
    tags=["focus"],
    summary="Сбросить фокус",
    description="Сбрасывает режим фокуса. Все узлы возвращаются к полной видимости.",
)
def api_focus_reset():
    global _sub_analyzer
    _get_analyzer()
    _sub_analyzer = None
    return {"status": "ok"}


@app.get(
    "/api/health",
    tags=["system"],
    summary="Проверка состояния",
    description="Возвращает `{\"status\": \"ok\"}`, если сервер работает.",
)
def health():
    return {"status": "ok"}
