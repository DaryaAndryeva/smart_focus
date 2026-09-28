from pydantic import BaseModel, Field
from enum import Enum


class GraphType(str, Enum):
    DENSE_GNM = "dense_gnm"
    RANDOM = "random"
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"
    POWERLAW_CLUSTER = "powerlaw_cluster"
    RANDOM_PARTITION = "random_partition"
    STOCHASTIC_BLOCK = "stochastic_block"
    BALANCED_TREE = "balanced_tree"
    INTERNET_AS = "internet_as"
    GRID = "grid"


class TopologyType(str, Enum):
    TREE = "tree"
    SPARSE = "sparse"
    CLUSTERED = "clustered"
    DENSE = "dense"
    SMALL_WORLD = "small_world"


class LayoutAlgorithm(str, Enum):
    SPRING = "spring"
    CIRCULAR = "circular"
    SPECTRAL = "spectral"
    KAMADA_KAWAI = "kamada_kawai"
    SHELL = "shell"
    COMMUNITY = "community"
    MULTILEVEL = "multilevel"


class GenerateRequest(BaseModel):
    """Параметры генерации графа."""

    graph_type: GraphType = Field(description="Тип генерируемого графа")
    n: int = Field(default=50, ge=5, le=100000, description="Количество вершин (5–100 000)")
    p: float = Field(default=0.1, ge=0.01, le=1.0, description="Вероятность ребра (для random, small_world)")
    m: int = Field(default=2, ge=1, le=20, description="Рёбер на новый узел (для scale_free)")
    k: int = Field(default=4, ge=2, le=20, description="Число соседей (для small_world)")
    num_edges: int | None = Field(default=None, ge=1, le=5000000, description="Количество рёбер (для dense_gnm)")
    auto_collapse: int | None = Field(default=None, ge=2, le=50000, description="Автоматическая свёртка: порог размера сообщества")


class FocusRequest(BaseModel):
    """Параметры вычисления контекста фокуса."""

    node_id: int = Field(description="ID узла для фокусировки")
    depth: int = Field(default=2, ge=1, le=5, description="Глубина BFS-обхода (1–5)")
    preserve_bridges: bool = Field(default=True, description="Сохранять видимость мостов (high betweenness)")
    preserve_hubs: bool = Field(default=True, description="Сохранять видимость хабов (high degree)")
    hub_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Перцентиль порога для хабов (0–1)")


class LayoutRequest(BaseModel):
    """Выбор алгоритма укладки."""

    algorithm: LayoutAlgorithm = Field(description="Алгоритм укладки графа")


class UploadRequest(BaseModel):
    """Загрузка графа по списку рёбер."""

    edges: list[list[int]] = Field(description="Список рёбер: [[source, target], ...]")


class NodeData(BaseModel):
    """Данные вершины графа с координатами и метриками."""

    id: int = Field(description="Уникальный идентификатор вершины")
    label: str = Field(description="Текстовая метка")
    x: float = Field(description="Нормализованная координата X (0–1)")
    y: float = Field(description="Нормализованная координата Y (0–1)")
    degree_centrality: float = Field(description="Центральность по степени")
    betweenness_centrality: float = Field(description="Центральность посредничества")
    closeness_centrality: float = Field(description="Центральность близости")
    eigenvector_centrality: float = Field(description="Собственная центральность")
    community: int = Field(description="Номер сообщества (алгоритм Лейдена)")
    degree: int = Field(description="Степень вершины")
    is_collapsed: bool = Field(default=False, description="Свёрнутое сообщество (мета-узел)")
    member_count: int = Field(default=1, description="Количество вершин в мета-узле")


class EdgeData(BaseModel):
    """Ребро графа."""

    source: int = Field(description="ID начальной вершины")
    target: int = Field(description="ID конечной вершины")
    weight: int = Field(default=1, description="Вес ребра (число агрегированных рёбер)")


class GlobalMetrics(BaseModel):
    """Глобальные метрики графа."""

    num_nodes: int = Field(description="Количество вершин")
    num_edges: int = Field(description="Количество рёбер")
    density: float = Field(description="Плотность графа (0–1)")
    diameter: int | None = Field(default=None, description="Диаметр (None для больших графов)")
    radius: int | None = Field(default=None, description="Радиус (None для больших графов)")
    clustering_coefficient: float = Field(description="Средний коэффициент кластеризации")
    num_components: int = Field(description="Число компонент связности")
    modularity: float = Field(description="Модулярность разбиения (алгоритм Лейдена)")
    avg_degree: float = Field(description="Средняя степень вершины")
    avg_path_length: float | None = Field(default=None, description="Средняя длина пути (None для больших графов)")


class GraphResponse(BaseModel):
    """Полный ответ с данными графа, метриками, топологией и укладкой."""

    nodes: list[NodeData] = Field(description="Список вершин с координатами и метриками")
    edges: list[EdgeData] = Field(description="Список рёбер")
    global_metrics: GlobalMetrics = Field(description="Глобальные метрики графа")
    topology: TopologyType = Field(description="Классифицированный тип топологии")
    recommended_layout: LayoutAlgorithm = Field(description="Рекомендованный алгоритм укладки")
    num_communities: int = Field(description="Количество обнаруженных сообществ")
    community_sizes: dict[int, int] = Field(description="Размеры сообществ {id: count}")


class CollapseRequest(BaseModel):
    """Параметры свёртки крупных сообществ."""

    min_size: int = Field(default=10, ge=2, le=5000, description="Минимальный размер сообщества для свёртки")


class SubgraphRequest(BaseModel):
    """Запрос подграфа одного сообщества."""

    community_id: int = Field(description="ID сообщества для просмотра")
